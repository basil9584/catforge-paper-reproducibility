# models/optimization/optimization_catcost.py
# CatCost-Enhanced Optimization Model with Full TEA Calculations
# Complete feature parity with optimization.py + enhanced cost analysis

import pandas as pd
import numpy as np
import os
import joblib
import json
import random
import optuna
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
import signal
import sys
import warnings
import unicodedata

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Reproducibility seeds
DEFAULT_SEED = 42
DEFAULT_N_TRIALS = 10

# --- TIMEOUT HANDLING ---
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Optimization timed out")

class OptimizationTimeout:
    def __init__(self, seconds=300):
        self.seconds = seconds
        self.is_windows = sys.platform == 'win32'
        
    def __enter__(self):
        if not self.is_windows:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.seconds)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_windows:
            signal.alarm(0)
        return False

NA_VALUES = ['-', 'NA', 'N/A', 'n/a', 'na', 'NaN', 'nan', '', ' ', 'null', 'NULL', 'None', 'none', '#N/A', '#NA']


# =============================================================================
# CATCOST TEA CALCULATION ENGINE
# =============================================================================

# Default CatCost factors (user can override via UI)
DEFAULT_CAPEX_FACTORS = {
    'direct': {
        'installation': {'baseline': 0.47, 'low': 0.35, 'high': 0.60},
        'instrumentation': {'baseline': 0.18, 'low': 0.10, 'high': 0.25},
        'piping': {'baseline': 0.66, 'low': 0.40, 'high': 0.80},
        'electrical': {'baseline': 0.11, 'low': 0.08, 'high': 0.15},
        'buildings': {'baseline': 0.18, 'low': 0.10, 'high': 0.25},
        'yard_improvements': {'baseline': 0.10, 'low': 0.05, 'high': 0.15},
        'service_facilities': {'baseline': 0.70, 'low': 0.50, 'high': 0.90},
    },
    'indirect': {
        'engineering': {'baseline': 0.33, 'low': 0.25, 'high': 0.45},
        'construction': {'baseline': 0.41, 'low': 0.30, 'high': 0.55},
        'legal_contractors': {'baseline': 0.04, 'low': 0.02, 'high': 0.06},
        'contingency': {'baseline': 0.10, 'low': 0.05, 'high': 0.15},
    },
    'other': {
        'working_capital': {'baseline': 0.15, 'low': 0.10, 'high': 0.20},
    }
}

DEFAULT_OPEX_FACTORS = {
    'direct': {
        'maintenance': {'baseline': 0.06, 'low': 0.04, 'high': 0.08, 'unit': '% of FCI'},
        'supplies': {'baseline': 0.15, 'low': 0.10, 'high': 0.20, 'unit': '% of M&R'},
        'lab_charges': {'baseline': 0.15, 'low': 0.10, 'high': 0.20, 'unit': '% of DL'},
    },
    'fixed': {
        'taxes': {'baseline': 0.02, 'low': 0.01, 'high': 0.03, 'unit': '% of FCI'},
        'insurance': {'baseline': 0.01, 'low': 0.005, 'high': 0.015, 'unit': '% of FCI'},
        'rent': {'baseline': 0.0, 'low': 0.0, 'high': 0.02, 'unit': '% of FCI'},
        'overhead': {'baseline': 0.50, 'low': 0.40, 'high': 0.60, 'unit': '% of LSM'},
    },
    'general': {
        'admin': {'baseline': 0.15, 'low': 0.10, 'high': 0.20, 'unit': '% of op costs'},
        'distribution': {'baseline': 0.05, 'low': 0.02, 'high': 0.10, 'unit': '% of op costs'},
        'rnd': {'baseline': 0.05, 'low': 0.02, 'high': 0.08, 'unit': '% of op costs'},
    }
}

def get_sensitivity_value(config, key='baseline'):
    """Safely extract value from sensitivity dict or return as-is."""
    if isinstance(config, dict):
        return float(config.get(key, config.get('baseline', 0)) or 0)
    return float(config) if config else 0.0


# The web interface (TeaEditor / CostRowBuilder) sends cost items as
# {name, costType: 'one_time'|'per_hour', cost, duration, durationUnit}; these are summed the same way the
# interface previews them (TIME_MULTIPLIERS matches OptimizationCatCost.jsx).
TIME_MULTIPLIERS = {'hours': 1, 'days': 24, 'weeks': 168}


def row_builder_total(items):
    """Total of interface cost items (one-time cost, or hourly cost x duration)."""
    total = 0.0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        cost = float(item.get('cost') or 0)
        if item.get('costType', 'one_time') == 'one_time':
            total += cost
        else:
            total += cost * float(item.get('duration') or 0) * TIME_MULTIPLIERS.get(item.get('durationUnit'), 1)
    return total


def calculate_materials_cost(materials_config, scale=1000):
    """
    Calculate total materials cost from CatCost-style config.
    
    Args:
        materials_config: List of material dicts with quantity/price sensitivity
        scale: Production scale (kg/year)
    
    Returns:
        dict with baseline, low, high costs and breakdown
    """
    total = {'baseline': 0, 'low': 0, 'high': 0}
    breakdown = []
    
    for mat in materials_config:
        name = mat.get('name', 'Unknown')
        # Accept both the CatCost format (quantity / unit_price) and the web interface format (qty / cost).
        q_raw = mat.get('quantity', mat.get('qty', 0))
        p_raw = mat.get('unit_price', mat.get('cost', 0))
        qty_b = get_sensitivity_value(q_raw, 'baseline')
        qty_l = get_sensitivity_value(q_raw, 'low') or qty_b
        qty_h = get_sensitivity_value(q_raw, 'high') or qty_b

        price_b = get_sensitivity_value(p_raw, 'baseline')
        price_l = get_sensitivity_value(p_raw, 'low') or price_b
        price_h = get_sensitivity_value(p_raw, 'high') or price_b
        
        cost_b = qty_b * price_b
        cost_l = qty_l * price_l  # Low cost = low qty × low price
        cost_h = qty_h * price_h  # High cost = high qty × high price
        
        total['baseline'] += cost_b
        total['low'] += cost_l
        total['high'] += cost_h
        
        breakdown.append({
            'name': name,
            'category': mat.get('category', 'other'),
            'cost_baseline': cost_b,
            'cost_low': cost_l,
            'cost_high': cost_h
        })
    
    return {
        'total': total,
        'breakdown': breakdown,
        'cost_per_kg': {
            'baseline': total['baseline'] / scale if scale > 0 else 0,
            'low': total['low'] / scale if scale > 0 else 0,
            'high': total['high'] / scale if scale > 0 else 0,
        }
    }


def calculate_equipment_cost(equipment_config):
    """
    Calculate total equipment purchase cost.
    
    Args:
        equipment_config: List of equipment dicts
    
    Returns:
        dict with total cost and breakdown
    """
    total = 0
    breakdown = []
    
    for equip in equipment_config:
        if 'base_cost' not in equip and 'cost' in equip:
            # Web interface format (CostRowBuilder item): purchase cost, or hourly cost x duration
            cost = row_builder_total([equip])
            total += cost
            breakdown.append({'name': equip.get('name') or 'Item', 'quantity': 1, 'unit_cost': cost, 'total_cost': cost})
            continue
        name = equip.get('equipment_type', 'Unknown')
        qty = float(equip.get('quantity', 1))
        base_cost = float(equip.get('base_cost', 0))
        size_factor = float(equip.get('size_factor', 1.0))
        material_factor = float(equip.get('material_factor', 1.0))
        
        cost = base_cost * qty * size_factor * material_factor
        total += cost
        
        breakdown.append({
            'name': name,
            'quantity': qty,
            'unit_cost': base_cost * size_factor * material_factor,
            'total_cost': cost
        })
    
    return {'total': total, 'breakdown': breakdown}


def calculate_capex(equipment_cost, capex_factors=None, project_life=10):
    """
    Calculate Capital Expenses using CatCost methodology.
    
    Args:
        equipment_cost: Total equipment purchase cost
        capex_factors: Dict of capital cost factors (% of equipment)
        project_life: Years for annualization
    
    Returns:
        dict with TCI, FCI, annualized cost, and breakdown
    """
    factors = capex_factors or DEFAULT_CAPEX_FACTORS
    
    # Direct Capital
    direct_costs = {'equipment_purchase': equipment_cost}
    for name, factor in factors.get('direct', {}).items():
        f = get_sensitivity_value(factor, 'baseline')
        direct_costs[name] = equipment_cost * f
    total_direct = sum(direct_costs.values())
    
    # Indirect Capital (% of direct)
    indirect_costs = {}
    for name, factor in factors.get('indirect', {}).items():
        f = get_sensitivity_value(factor, 'baseline')
        indirect_costs[name] = total_direct * f
    total_indirect = sum(indirect_costs.values())
    
    # Fixed Capital Investment
    fci = total_direct + total_indirect
    
    # Working Capital & TCI
    wc_factor = get_sensitivity_value(factors.get('other', {}).get('working_capital', 0.15), 'baseline')
    working_capital = fci * wc_factor
    tci = fci + working_capital
    
    # Annualized (simple straight-line)
    annualized = tci / project_life if project_life > 0 else tci
    
    return {
        'equipment_purchase': equipment_cost,
        'total_direct': total_direct,
        'total_indirect': total_indirect,
        'fci': fci,
        'working_capital': working_capital,
        'tci': tci,
        'annualized': annualized,
        'direct_breakdown': direct_costs,
        'indirect_breakdown': indirect_costs
    }


def calculate_opex(fci, materials_cost, labor_config, opex_factors=None, scale=1000, extra_items=None):
    """
    Calculate Operating Expenses using CatCost methodology.
    
    Args:
        fci: Fixed Capital Investment
        materials_cost: Annual raw materials cost
        labor_config: Dict with rate, operators, hours
        opex_factors: Dict of operating cost factors
        scale: Production scale (kg/year)
    
    Returns:
        dict with annual OpEx and breakdown
    """
    factors = opex_factors or DEFAULT_OPEX_FACTORS
    extra_items = extra_items or {}

    # Direct Labor (the interface may send labour as a list of cost items instead of rate/operators/hours)
    if isinstance(labor_config, list):
        direct_labor = row_builder_total(labor_config)
        labor_config = {}
    else:
        labor_config = labor_config or {}
        labor_rate = float(labor_config.get('rate', 25))  # $/hr
        operators = float(labor_config.get('operators', 2))
        hours_per_year = float(labor_config.get('hours_per_year', 2080))
        direct_labor = labor_rate * operators * hours_per_year
    
    # Maintenance & Repair
    m_and_r = fci * get_sensitivity_value(factors['direct'].get('maintenance', 0.06), 'baseline')
    
    # Direct Operating Costs
    direct_costs = {
        'raw_materials': materials_cost,
        'direct_labor': direct_labor,
        'maintenance_repair': m_and_r,
        'supplies': m_and_r * get_sensitivity_value(factors['direct'].get('supplies', 0.15), 'baseline'),
        'lab_charges': direct_labor * get_sensitivity_value(factors['direct'].get('lab_charges', 0.15), 'baseline'),
    }
    
    # Utilities (simplified - user can configure), plus interface cost items for utilities, lab and ancillary costs
    utilities = float(labor_config.get('utilities_annual', 0)) + row_builder_total(extra_items.get('utility_cost'))
    direct_costs['utilities'] = utilities
    direct_costs['lab_items'] = row_builder_total(extra_items.get('lab_cost'))
    direct_costs['ancillary'] = row_builder_total(extra_items.get('ancillary_cost'))
    
    lsm_total = sum(direct_costs.values())  # Labor, Supplies, Maintenance total
    
    # Fixed/Indirect Operating Costs
    fixed_costs = {
        'taxes': fci * get_sensitivity_value(factors['fixed'].get('taxes', 0.02), 'baseline'),
        'insurance': fci * get_sensitivity_value(factors['fixed'].get('insurance', 0.01), 'baseline'),
        'rent': fci * get_sensitivity_value(factors['fixed'].get('rent', 0), 'baseline'),
        'overhead': lsm_total * get_sensitivity_value(factors['fixed'].get('overhead', 0.50), 'baseline'),
    }
    
    subtotal = lsm_total + sum(fixed_costs.values())
    
    # General Expenses
    general_costs = {
        'admin': subtotal * get_sensitivity_value(factors['general'].get('admin', 0.15), 'baseline'),
        'distribution': subtotal * get_sensitivity_value(factors['general'].get('distribution', 0.05), 'baseline'),
        'rnd': subtotal * get_sensitivity_value(factors['general'].get('rnd', 0.05), 'baseline'),
    }
    
    total_opex = subtotal + sum(general_costs.values())
    
    return {
        'total_annual': total_opex,
        'cost_per_kg': total_opex / scale if scale > 0 else 0,
        'direct': direct_costs,
        'fixed': fixed_costs,
        'general': general_costs,
        'lsm_total': lsm_total
    }


def calculate_spent_catalyst(spent_config, scale=1000):
    """
    Calculate Spent Catalyst end-of-life value/cost.
    
    Args:
        spent_config: Dict with recovery rates, disposal costs, metal content
        scale: Production scale (kg/year)
    
    Returns:
        dict with net cost/credit
    """
    if not spent_config:
        return {'net_annual': 0, 'net_per_kg': 0}

    if 'massContent' in spent_config:
        # Web interface format: SCV = metal mass x recovery efficiency (%) x metal value - recovery cost
        credit = (float(spent_config.get('massContent') or 0) * float(spent_config.get('efficiency') or 0) / 100
                  * float(spent_config.get('metalValue') or 0) - float(spent_config.get('recoveryCost') or 0))
        return {'recovery_credit': credit, 'disposal_cost': 0, 'net_annual': -credit,
                'net_per_kg': -credit / scale if scale > 0 else 0}

    # Metal recovery credit
    metal_content_kg = float(spent_config.get('metal_content_kg', 0))
    metal_value_per_kg = float(spent_config.get('metal_value', 0))
    recovery_rate = float(spent_config.get('recovery_rate', 0.9))
    refining_charge = float(spent_config.get('refining_charge', 0))
    
    recovery_credit = metal_content_kg * metal_value_per_kg * recovery_rate - refining_charge
    
    # Disposal cost
    spent_mass_kg = float(spent_config.get('spent_mass_kg', 0))
    disposal_cost_per_kg = float(spent_config.get('disposal_cost', 0))
    hazard_factor = float(spent_config.get('hazard_factor', 1.0))
    
    disposal_cost = spent_mass_kg * disposal_cost_per_kg * hazard_factor * (1 - recovery_rate)
    
    # Net (negative = credit, positive = cost)
    net = disposal_cost - recovery_credit
    
    return {
        'recovery_credit': recovery_credit,
        'disposal_cost': disposal_cost,
        'net_annual': net,
        'net_per_kg': net / scale if scale > 0 else 0
    }


def calculate_total_tea(tea_config, scale=1000):
    """
    Master TEA calculation combining all modules.
    
    Args:
        tea_config: Full configuration dict with all modules
        scale: Production scale (kg/year)
    
    Returns:
        Complete cost breakdown with unit cost
    """
    # 1. Materials
    materials = calculate_materials_cost(
        tea_config.get('materials', []),
        scale
    )
    materials_annual = materials['total']['baseline']
    
    # 2. Equipment & CapEx
    equipment = calculate_equipment_cost(tea_config.get('equipment', []))
    capex = calculate_capex(
        equipment['total'],
        tea_config.get('capex_factors'),
        tea_config.get('project_life', 10)
    )
    
    # 3. OpEx
    opex = calculate_opex(
        capex['fci'],
        materials_annual,
        tea_config.get('labor', {}),
        tea_config.get('opex_factors'),
        scale,
        extra_items={k: tea_config.get(k) for k in ('utility_cost', 'lab_cost', 'ancillary_cost')}
    )
    
    # 4. Spent Catalyst
    spent = calculate_spent_catalyst(tea_config.get('spent_catalyst', {}), scale)
    
    # 5. Total Unit Cost
    total_annual = capex['annualized'] + opex['total_annual'] + spent['net_annual']
    unit_cost = total_annual / scale if scale > 0 else 0
    
    return {
        'scale': scale,
        'materials': materials,
        'equipment': equipment,
        'capex': capex,
        'opex': opex,
        'spent_catalyst': spent,
        'total_annual_cost': total_annual,
        'unit_cost_per_kg': unit_cost,
        'cost_breakdown_percent': {
            'materials': (materials_annual / total_annual * 100) if total_annual > 0 else 0,
            'capex': (capex['annualized'] / total_annual * 100) if total_annual > 0 else 0,
            'opex': (opex['total_annual'] / total_annual * 100) if total_annual > 0 else 0,
            'spent_cat': (spent['net_annual'] / total_annual * 100) if total_annual > 0 else 0,
        }
    }


# =============================================================================
# PER-RECORD / PER-CANDIDATE COST (composition-linked materials)
# =============================================================================
# A material can be linked to an input column so that its quantity follows each record's composition:
#   {"name": "Ni (metal)", "cost": 17.0, "link_column": "Ni Loading", "link_factor": 0.01, "link_offset": 0}
#       -> kg of material per kg of catalyst = link_offset + link_factor * value of "Ni Loading" (wt% -> 0.01 x wt%)
#   {"name": "Al2O3 support", "cost": 0.57, "balance": true}
#       -> kg per kg catalyst = 1 - sum(fraction_in_catalyst x kg of every linked material)
# Optional "fraction_in_catalyst" (default 1) is the mass of a linked material that ends up in the catalyst
# (for example 0.202 for Ni(NO3)2.6H2O when only Ni remains). Linked quantities are converted to annual
# amounts at the configured scale, and the full CatCost-style TEA is evaluated for each record or candidate.

def _row_number(row, col):
    """Numeric value of column `col` in a record (tolerates surrounding spaces in column names)."""
    keys = [col] + [k for k in row.keys() if isinstance(k, str) and k.strip() == str(col).strip()]
    for k in keys:
        if k in row:
            try:
                v = float(row[k])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
    return None


def _num(value, default):
    """Number from an interface field ('' or missing -> default; 0 stays 0)."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def has_linked_materials(tea_config):
    return bool(tea_config) and any(isinstance(m, dict) and (m.get('link_column') or m.get('balance'))
                                    for m in tea_config.get('materials', []))


def compute_row_tea_cost(tea_config, row):
    """Unit cost (per kg catalyst) for one record or candidate, with linked material quantities from the record.
    Returns None when a linked column is missing or not numeric in this record."""
    cfg = row_tea_config(tea_config, row)
    if cfg is None:
        return None
    return calculate_total_tea(cfg, cfg.get('scale', 1000))['unit_cost_per_kg']


def row_tea_config(tea_config, row):
    """Copy of the TEA configuration with linked and balance material quantities set from one record."""
    scale = float(tea_config.get('scale', 1000)) or 1000.0
    mats = [dict(m) for m in tea_config.get('materials', [])]
    per_kg, in_catalyst = {}, 0.0
    for i, m in enumerate(mats):
        col = m.get('link_column')
        if not col:
            continue
        v = _row_number(row, col)
        if v is None:
            return None
        q = _num(m.get('link_offset'), 0.0) + _num(m.get('link_factor'), 1.0) * v
        per_kg[i] = max(q, 0.0)
        in_catalyst += per_kg[i] * _num(m.get('fraction_in_catalyst'), 1.0)
    for i, m in enumerate(mats):
        if i in per_kg:
            q = per_kg[i]
        elif m.get('balance'):
            q = max(0.0, 1.0 - in_catalyst)
        else:
            continue
        m['quantity'] = q * scale            # annual kg at the configured production scale
        m.pop('qty', None)
    cfg = dict(tea_config); cfg['materials'] = mats
    return cfg


def row_unit_cost(job_dir, config, tea_config, row, base_unit_cost, row_index=None):
    """Cost per kg used for ranking. Priority: composition-linked materials; then per-catalyst recipes
    (catalyst_config.json, base optimizer); then the job's unit cost. For measured records, a per-record
    override entered in the interface (row_cost_config) scales the result by its ratio to the global total."""
    cost = None
    if has_linked_materials(tea_config):
        cost = compute_row_tea_cost(tea_config, row)
        if cost is None and tea_config.get('linked_fallback_unit_cost') is not None:
            # Record without a usable value in a linked column: use the median record's cost
            base_unit_cost = float(tea_config['linked_fallback_unit_cost'])
    if cost is None:
        cost = compute_row_catalyst_cost(job_dir, config, row, base_unit_cost)
    if row_index is not None and tea_config:
        overrides = tea_config.get('row_cost_config') or {}
        ref = float(tea_config.get('row_cost_reference_total') or 0)
        entry = overrides.get(str(row_index), overrides.get(row_index)) if isinstance(overrides, dict) else None
        if entry and ref > 0:
            try:
                total = float(entry.get('total') if isinstance(entry, dict) else entry)
                if np.isfinite(total) and abs(total - ref) > 1e-9:
                    cost = cost * total / ref
            except (TypeError, ValueError):
                pass
    return cost


# =============================================================================
# OPTIMIZATION WORKFLOW STEPS (Enhanced from base optimization.py)
# =============================================================================

def clean_catalyst_names(s):
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize('NFKD', s)
    s = s.replace('\xa0', ' ').replace('–', '-').replace('—', '-')
    return s.strip()

def generalize_clean_text(value):
    if pd.isna(value) or not isinstance(value, str):
        return value
    value = unicodedata.normalize('NFKD', value)
    value = value.replace('\xa0', ' ').replace('–', '-').replace('—', '-')
    return ' '.join(value.split())

def compute_row_catalyst_cost(job_dir, config, row, fallback):
    """
    Computes a PER-ROW running cost from the row's own catalyst columns using
    catalyst_config.json (same lookup the base optimization.py step 6 uses).
    Returns `fallback` when no catalyst profile matches or the config is missing.
    """
    try:
        with open(os.path.join(job_dir, 'catalyst_config.json')) as f:
            cat_conf = json.load(f)
    except Exception:
        cat_conf = {}
    if not cat_conf:
        return fallback
    catalyst_cols = config.get('catalyst_cols', [])
    for cid, comps in cat_conf.items():
        clean_cid = generalize_clean_text(str(cid))
        if any(generalize_clean_text(str(row.get(col, ''))) == clean_cid for col in catalyst_cols):
            cost = sum(float(x.get('mass', 0)) / 1000 * float(x.get('cost', 0)) for x in comps)
            if cost > 0:
                return cost
    return fallback


def step1_upload_and_analyze(file_path, job_dir):
    """Upload and analyze data - same as base optimizer."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, thousands=',', na_values=NA_VALUES)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, na_values=NA_VALUES)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or XLSX.")
    
    df.columns = df.columns.str.strip()
    # Only drop completely empty rows, not rows with some NA values
    df = df.dropna(how='all')
    df.to_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'), index=False)
    return df.columns.tolist()


def step2_configure_and_prepare(job_id, job_dir, config):
    """Configure inputs/targets - same as base optimizer with active learning support."""
    df = None
    
    # Support for patched data
    if 'patched_data' in config and config['patched_data']:
        try:
            df = pd.DataFrame(config['patched_data'])
            df.to_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'), index=False)
        except:
            df = None
    
    if df is None:
        df = pd.read_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'), na_values=NA_VALUES)
    
    # Active Learning: Append new rows from lab experiments
    new_rows = config.get('new_rows', [])
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'), index=False)
        print(f"DEBUG: Appended {len(new_rows)} new rows via active learning")

    
    print(f"DEBUG: Loaded data shape = {df.shape}")
    print(f"DEBUG: Columns = {df.columns.tolist()}")
    
    input_cols = config['inputs']
    target_cols = config['targets']
    categorical_cols = config.get('categoricals', [])
    
    print(f"DEBUG: input_cols = {input_cols}, target_cols = {target_cols}")
    
    relevant_cols = input_cols + target_cols
    # Check which columns actually exist
    existing_cols = [c for c in relevant_cols if c in df.columns]
    print(f"DEBUG: existing relevant cols = {existing_cols}")
    
    if len(existing_cols) < len(relevant_cols):
        missing = [c for c in relevant_cols if c not in df.columns]
        print(f"WARNING: Missing columns: {missing}")
    
    # Only drop rows where ALL relevant columns are NA, not any single one
    df_before = len(df)
    df = df.dropna(subset=existing_cols, how='all')
    print(f"DEBUG: After dropna(how='all'): {df_before} -> {len(df)} rows")
    
    # Fill remaining NAs with column median for numeric columns
    # NOTE: user-marked categorical columns are NEVER coerced to numeric — mirroring the
    # base module's strict separation of numeric vs categorical handling.
    for col in existing_cols:
        if col in df.columns:
            if col in categorical_cols:
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).fillna("Unknown")
                continue
            if df[col].dtype in ['float64', 'int64']:
                na_count = df[col].isna().sum()
                if na_count > 0:
                    df[col] = df[col].fillna(df[col].median())
                    print(f"DEBUG: Filled {na_count} NAs in {col}")
            else:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
    
    if len(df) == 0:
        raise ValueError(f"No valid data remaining. Original shape was {df_before} rows.")
    
    X = df[input_cols].copy()
    Y = df[target_cols].copy()
    
    if categorical_cols:
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    else:
        X_encoded = X
    
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    for col in numeric_cols:
        if col in X_encoded.columns:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
    
    for col in Y.columns:
        Y[col] = pd.to_numeric(Y[col], errors='coerce')
    
    combined = pd.concat([X_encoded, Y], axis=1).dropna()
    X_encoded = combined[X_encoded.columns]
    Y = combined[Y.columns]
    
    scaler = StandardScaler()
    valid_numeric_cols = [c for c in numeric_cols if c in X_encoded.columns]
    if valid_numeric_cols:
        X_encoded[valid_numeric_cols] = scaler.fit_transform(X_encoded[valid_numeric_cols])
    
    X_encoded.to_csv(os.path.join(job_dir, 'step2_X_encoded.csv'), index=False)
    Y.to_csv(os.path.join(job_dir, 'step2_Y.csv'), index=False)
    joblib.dump(scaler, os.path.join(job_dir, 'scaler.pkl'))
    joblib.dump(config, os.path.join(job_dir, 'config.json'))
    
    return {'message': 'Configuration successful'}


def step3_configure_tea(job_id, job_dir, tea_config):
    """
    NEW: Save full TEA configuration including CatCost-style modules.
    """
    # Merge with defaults for any missing sections
    full_config = {
        'materials': tea_config.get('materials', []),
        'equipment': tea_config.get('equipment', []),
        'labor': tea_config.get('labor', {
            'rate': 25,
            'operators': 2,
            'hours_per_year': 2080,
            'utilities_annual': 5000
        }),
        'capex_factors': tea_config.get('capex_factors', DEFAULT_CAPEX_FACTORS),
        'opex_factors': tea_config.get('opex_factors', DEFAULT_OPEX_FACTORS),
        'spent_catalyst': tea_config.get('spent_catalyst', {}),
        'project_life': tea_config.get('project_life', 10),
        'scale': tea_config.get('scale', 1000),
        # Interface cost items and per-record overrides (previously dropped here)
        'utility_cost': tea_config.get('utility_cost', []),
        'lab_cost': tea_config.get('lab_cost', []),
        'ancillary_cost': tea_config.get('ancillary_cost', []),
        'row_cost_config': tea_config.get('row_cost_config', {}),
        'row_cost_reference_total': tea_config.get('row_cost_reference_total'),
        # Weight of cost in the TOPSIS ranking (None = mean of the response weights, the previous behaviour)
        'cost_weight': tea_config.get('cost_weight'),
    }
    
    # Composition-linked materials: check the links and evaluate the cost of every uploaded record
    warnings, cost_range, preview_cfg = [], None, full_config
    if has_linked_materials(full_config):
        try:
            inputs = joblib.load(os.path.join(job_dir, 'config.json')).get('inputs', [])
        except Exception:
            inputs = []
        for m in full_config['materials']:
            col = m.get('link_column') if isinstance(m, dict) else None
            if col and inputs and col not in inputs:
                warnings.append(f"'{m.get('name') or col}' is linked to '{col}', which is not a model input: "
                                f"suggested candidates will use the median record's cost.")
        try:
            df = pd.read_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'), na_values=NA_VALUES)
            rows = [r for _, r in df.iterrows()]
            costs = [compute_row_tea_cost(full_config, r) for r in rows]
            valid = [(c, r) for c, r in zip(costs, rows) if c is not None and np.isfinite(c)]
            if valid:
                vals = np.array([c for c, _ in valid])
                med_i = int(np.argmin(np.abs(vals - np.median(vals))))
                preview_cfg = row_tea_config(full_config, valid[med_i][1])
                full_config['linked_fallback_unit_cost'] = float(vals[med_i])
                cost_range = {'min': round(float(vals.min()), 4), 'median': round(float(np.median(vals)), 4),
                              'max': round(float(vals.max()), 4), 'n_records': int(len(vals)),
                              'n_missing': int(len(rows) - len(vals))}
            else:
                warnings.append('No uploaded record has numeric values in the linked columns.')
        except Exception as e:
            warnings.append(f'Per-record cost preview failed: {e}')

    with open(os.path.join(job_dir, 'tea_config.json'), 'w') as f:
        json.dump(full_config, f, indent=2)

    # Calculate preview cost (for linked materials: the median record)
    preview = calculate_total_tea(preview_cfg, full_config['scale'])

    return {
        'message': 'TEA configuration saved.',
        'warnings': warnings,
        'preview': {
            'unit_cost_per_kg': round(preview['unit_cost_per_kg'], 4),
            'total_annual_cost': round(preview['total_annual_cost'], 2),
            'cost_breakdown_percent': preview['cost_breakdown_percent'],
            'per_record_cost': cost_range
        }
    }


def step4_filter_data(job_id, job_dir, filter_config):
    """Apply filters and calculate TEA costs for each row."""
    df = pd.read_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'), na_values=NA_VALUES)
    config = joblib.load(os.path.join(job_dir, 'config.json'))
    
    # Load TEA config
    try:
        with open(os.path.join(job_dir, 'tea_config.json')) as f:
            tea_config = json.load(f)
    except:
        tea_config = None
    
    # Calculate cost if TEA configured
    if tea_config:
        scale = tea_config.get('scale', 1000)
        tea_result = calculate_total_tea(tea_config, scale)
        # Per-row cost: match each row's catalyst columns against catalyst_config.json
        # so cost varies by row instead of assigning one constant to every row. Falls
        # back to the global TEA unit cost when no per-catalyst profile matches.
        costs = [
            row_unit_cost(job_dir, config, tea_config, row, tea_result['unit_cost_per_kg'], row_index=idx)
            for idx, row in df.iterrows()
        ]
        df['Total_Running_Cost'] = costs
        df['Cost_Breakdown'] = json.dumps(tea_result['cost_breakdown_percent'])
    else:
        df['Total_Running_Cost'] = 0.0
        df['Cost_Breakdown'] = '{}'
    
    # Apply filters
    active_filters = [f for f in filter_config.get('filters', []) if f.get('column') and f.get('value')]
    if active_filters:
        logic = filter_config.get('logic', 'AND')
        mask = pd.Series([logic == 'AND'] * len(df), index=df.index)
        
        for f in active_filters:
            col, op, val = f['column'], f['operator'], f['value']
            try:
                numeric_val = float(val)
                col_vals = pd.to_numeric(df[col], errors='coerce')
                
                if op == '>': m = col_vals > numeric_val
                elif op == '<': m = col_vals < numeric_val
                elif op == '>=': m = col_vals >= numeric_val
                elif op == '<=': m = col_vals <= numeric_val
                elif op == '==': m = np.isclose(col_vals, numeric_val)
                elif op == '!=': m = ~np.isclose(col_vals, numeric_val)
                else: m = pd.Series([False] * len(df), index=df.index)
            except:
                s_vals = df[col].astype(str)
                if op == '==': m = s_vals == val
                elif op == '!=': m = s_vals != val
                else: m = pd.Series([False] * len(df), index=df.index)
            
            if logic == 'AND':
                mask &= m.fillna(False)
            else:
                mask |= m.fillna(False)
        
        df = df[mask]
    
    df.to_csv(os.path.join(job_dir, 'step4_filtered_data.csv'), index=False)
    return df


def step5_rank_data(job_id, job_dir):
    """Rank data based on weighted scores."""
    df = pd.read_csv(os.path.join(job_dir, 'step4_filtered_data.csv'), na_values=NA_VALUES)
    config = joblib.load(os.path.join(job_dir, 'config.json'))
    
    scores = np.zeros(len(df))
    for t in config['targets']:
        w = float(config['weights'].get(t, 0))
        vals = pd.to_numeric(df[t], errors='coerce').fillna(0)
        
        # Min-max normalization (matches the base module's step 5 behavior)
        max_val = vals.max()
        min_val = vals.min()
        if max_val > min_val:
            norm = (vals - min_val) / (max_val - min_val)
        else:
            norm = vals * 0.0 + 0.5
        
        if config.get('objectives', {}).get(t) == 'minimize':
            norm = 1.0 - norm
        
        scores += norm * w
    
    df['Score'] = scores
    df.sort_values('Score', ascending=False, inplace=True)
    df.to_csv(os.path.join(job_dir, 'step5_ranked_samples.csv'), index=False)
    return df


# --- STEP 6: "DEEP" TRAINING & GENERATION ---

def step6_suggest_candidates(job_id, job_dir, seed=DEFAULT_SEED):
    """Generate AI candidates with TEA cost estimation."""
    random.seed(seed)
    np.random.seed(seed)

    job_dir = str(job_dir)
    
    try:
        X_train = pd.read_csv(os.path.join(job_dir, 'step2_X_encoded.csv'))
        Y_train = pd.read_csv(os.path.join(job_dir, 'step2_Y.csv'))
        df_orig = pd.read_csv(os.path.join(job_dir, 'step1_uploaded_data.csv'))
        config = joblib.load(os.path.join(job_dir, 'config.json'))
        
        try:
            with open(os.path.join(job_dir, 'tea_config.json')) as f:
                tea_config = json.load(f)
        except:
            tea_config = None
        
        try:
            scaler = joblib.load(os.path.join(job_dir, 'scaler.pkl'))
        except:
            scaler = None
    except Exception as e:
        return {'error': str(e)}
    
    targets = config['targets']
    objectives = config.get('objectives', {t: 'maximize' for t in targets})
    input_cols = config['inputs']
    numeric_cols = [c for c in input_cols if c not in config.get('categoricals', [])]
    categorical_cols = config.get('categoricals', [])
    feature_constraints = config.get('feature_constraints', {})
    
    def get_col_values(df, col_name):
        if col_name in df.columns:
            return df[col_name]
        for c in df.columns:
            if c.strip() == col_name.strip():
                return df[c]
        return pd.Series(dtype=float)
    
    def get_safe_bounds(col_name):
        series = pd.to_numeric(get_col_values(df_orig, col_name), errors='coerce').dropna()
        if not series.empty:
            raw_min, raw_max = float(series.min()), float(series.max())
            span = raw_max - raw_min if raw_max > raw_min else 1.0
            data_min = raw_min - (0.10 * span)
            data_max = raw_max + (0.10 * span)
        else:
            data_min, data_max = 0.0, 1.0
        
        user_c = feature_constraints.get(col_name, {})
        if str(user_c.get('min', '')).strip():
            try:
                data_min = float(user_c['min'])
            except:
                pass
        if str(user_c.get('max', '')).strip():
            try:
                data_max = float(user_c['max'])
            except:
                pass
        
        if data_min >= data_max:
            data_max = data_min + 1e-6
        
        return data_min, data_max
    
    # Train models
    model_ensembles = {}
    for t in targets:
        models = []
        y = Y_train[t].values
        X = X_train.values
        
        for i in range(10):
            rng = np.random.default_rng(seed + i)
            indices = rng.choice(len(X), size=len(X), replace=True)
            m = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42+i
            )
            m.fit(X[indices], y[indices])
            models.append({'med': m})
        model_ensembles[t] = models
    
    # Generate candidates
    candidates = []
    n_samples = len(df_orig)
    
    if n_samples < 30:
        # Local optimization
        df_orig['Rank_Score'] = 0
        for t in targets:
            vals = pd.to_numeric(get_col_values(df_orig, t), errors='coerce')
            if objectives.get(t) == 'minimize':
                norm_v = 1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            else:
                norm_v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            df_orig['Rank_Score'] += norm_v
        
        elites = df_orig.sort_values('Rank_Score', ascending=False).head(5).to_dict('records')
        
        for parent in elites:
            for _ in range(6):
                child = parent.copy()
                for col in numeric_cols:
                    val = pd.to_numeric(parent.get(col, 0), errors='coerce')
                    if pd.isna(val):
                        val = 0
                    if random.random() > 0.3:
                        val = val * (1 + random.uniform(-0.10, 0.10))
                    c_min, c_max = get_safe_bounds(col)
                    child[col] = max(c_min, min(c_max, val))
                candidates.append(child)
    else:
        # Global optimization
        def objective(trial):
            params = {}
            for col in numeric_cols:
                low, high = get_safe_bounds(col)
                params[col] = trial.suggest_float(col, low, high)
            for col in categorical_cols:
                opts = get_col_values(df_orig, col).dropna().astype(str).unique().tolist()
                params[col] = trial.suggest_categorical(col, opts if opts else ['Unknown'])
            return random.random()
        
        try:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=DEFAULT_N_TRIALS, n_jobs=1)
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    candidates.append(trial.params)
        except Exception as e:
            print(f"Global Search Optuna Error: {e}")
    
    # Score candidates
    final_rows = []
    for row in candidates:
        try:
            df_trial = pd.DataFrame([row])
            if categorical_cols:
                df_enc = pd.get_dummies(df_trial, columns=categorical_cols, drop_first=False)
            else:
                df_enc = df_trial
            
            df_enc = df_enc.reindex(columns=X_train.columns, fill_value=0)
            
            if scaler:
                v_cols = [c for c in numeric_cols if c in df_enc.columns]
                if v_cols:
                    try:
                        df_enc[v_cols] = scaler.transform(df_enc[v_cols])
                    except:
                        pass
            
            total_score = 0
            for t in targets:
                ensemble = model_ensembles.get(t, [])
                if not ensemble:
                    continue
                
                preds = [m['med'].predict(df_enc)[0] for m in ensemble]
                mu = np.mean(preds)
                sigma = np.std(preds)
                
                row[f"{t}_pred"] = mu
                row[f"{t}_uncertainty"] = sigma
                
                max_v = pd.to_numeric(get_col_values(df_orig, t), errors='coerce').max()
                norm = mu / max_v if max_v else 0
                if objectives.get(t) == 'minimize':
                    norm = 1 - norm
                
                total_score += norm
            
            row['Predicted_Score'] = total_score
            
            # Calculate TEA cost (per-row: vary by the candidate's catalyst column value,
            # falling back to the global unit cost when no catalyst profile matches)
            if tea_config:
                tea_result = calculate_total_tea(tea_config, tea_config.get('scale', 1000))
                row['Total_Running_Cost'] = row_unit_cost(job_dir, config, tea_config, row, tea_result['unit_cost_per_kg'])
            else:
                row['Total_Running_Cost'] = 0
            
            final_rows.append(row)
        except:
            continue
    
    df_res = pd.DataFrame(final_rows)
    if not df_res.empty:
        df_res = df_res.sort_values('Predicted_Score', ascending=False).head(30)
    
    df_res.to_csv(os.path.join(job_dir, 'step6_suggested_candidates.csv'), index=False)
    
    # === GENERATE VISUALIZATION DATA ===
    viz_data = {}
    try:
        X_history = pd.read_csv(os.path.join(job_dir, 'step2_X_encoded.csv'))
        
        if not df_res.empty:
            # 1. ALIGN CANDIDATES (ONE-HOT ENCODING)
            if categorical_cols:
                cand_encoded = pd.get_dummies(df_res[input_cols], columns=categorical_cols)
            else:
                cand_encoded = df_res[input_cols].copy()
            
            X_cand_aligned = cand_encoded.reindex(columns=X_history.columns, fill_value=0)
            
            for c in X_cand_aligned.columns:
                X_cand_aligned[c] = pd.to_numeric(X_cand_aligned[c], errors='coerce').fillna(0)
            
            # 2. SCALING
            X_history_raw = X_history.copy()
            X_cand_scaled = X_cand_aligned.copy()
            
            if scaler and numeric_cols:
                valid_numerics = [c for c in numeric_cols if c in X_history.columns]
                try:
                    X_history_raw[valid_numerics] = scaler.inverse_transform(X_history[valid_numerics])
                except Exception as ex:
                    print(f"Viz Warning: Could not inverse transform history: {ex}")
                try:
                    X_cand_scaled[valid_numerics] = scaler.transform(X_cand_aligned[valid_numerics])
                except Exception as ex:
                    print(f"Viz Warning: Could not scale candidates: {ex}")
            
            # 3. PCA MANIFOLD
            X_combined_scaled = np.vstack([X_history, X_cand_scaled])
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_combined_scaled)
            
            n_hist = len(X_history)
            manifold_points = []
            
            for i in range(n_hist):
                manifold_points.append({
                    'x': float(coords[i, 0]),
                    'y': float(coords[i, 1]),
                    'type': 'History',
                    'score': 0
                })
            
            for i in range(len(df_res)):
                manifold_points.append({
                    'x': float(coords[n_hist + i, 0]),
                    'y': float(coords[n_hist + i, 1]),
                    'type': 'Candidate',
                    'score': float(df_res.iloc[i].get('Predicted_Score', 0))
                })
            
            viz_data['manifold'] = manifold_points
            
            # 4. LANDSCAPE (Feature Importance)
            avg_importance = {}
            for t, models in model_ensembles.items():
                if not models:
                    continue
                try:
                    imps = models[0]['med'].feature_importances_
                    for idx, col in enumerate(X_history.columns):
                        avg_importance[col] = avg_importance.get(col, 0) + imps[idx]
                except:
                    pass
            
            sorted_feats = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_feats[:2]]
            
            if len(top_features) >= 2:
                f1, f2 = top_features[0], top_features[1]
                landscape_points = []
                
                for i, row in X_history_raw.iterrows():
                    landscape_points.append({
                        'x': float(row.get(f1, 0)),
                        'y': float(row.get(f2, 0)),
                        'type': 'History',
                        'val': 0.5
                    })
                
                for i, row in X_cand_aligned.iterrows():
                    landscape_points.append({
                        'x': float(row.get(f1, 0)),
                        'y': float(row.get(f2, 0)),
                        'type': 'Candidate',
                        'val': float(df_res.iloc[i].get('Predicted_Score', 1.0))
                    })
                
                viz_data['landscape'] = {
                    'x_label': f1,
                    'y_label': f2,
                    'points': landscape_points
                }
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    return {
        'candidates': df_res.replace({np.nan: None}).to_dict(orient='records'),
        'headers': df_res.columns.tolist(),
        'visualization': viz_data
    }


# =========================================================
# HELPER: TOPSIS RANKING (Multi-Criteria Decision Making)
# =========================================================
def calculate_topsis_score(df, criteria_cols, weights, objectives):
    """
    Calculates the TOPSIS score for ranking candidates.
    TOPSIS measures the geometric distance to the 'Ideal Best' and 'Ideal Worst'.
    """
    # 1. Extract and Normalize Matrix
    valid_cols = [c for c in criteria_cols if c in df.columns]
    if not valid_cols:
        return np.array([0.0] * len(df))
        
    data = df[valid_cols].values.astype(float)
    
    if len(data) == 0:
        return np.array([])
        
    norm = np.sqrt((data**2).sum(axis=0))
    norm[norm == 0] = 1.0 
    normalized_data = data / norm
    
    # 2. Apply Weights
    weight_vector = np.array([weights.get(c, 1.0) for c in valid_cols])
    weighted_data = normalized_data * weight_vector
    
    # 3. Determine Ideal Best and Ideal Worst
    ideal_best = []
    ideal_worst = []
    
    for i, col in enumerate(valid_cols):
        direction = objectives.get(col, 'maximize')
        col_data = weighted_data[:, i]
        
        if direction == 'maximize':
            ideal_best.append(col_data.max())
            ideal_worst.append(col_data.min())
        else:  # minimize
            ideal_best.append(col_data.min())
            ideal_worst.append(col_data.max())
            
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)
    
    # 4. Calculate Distances
    dist_to_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))
    
    # 5. Calculate Score
    total_dist = dist_to_best + dist_to_worst
    total_dist[total_dist == 0] = 1.0 
    
    scores = dist_to_worst / total_dist
    return scores


# =========================================================
# STEP 7: ADVANCED PARETO & TECHNO-ECONOMICS
# =========================================================

def step7_pareto_analysis(job_id, job_dir):
    """Pareto analysis with enhanced cost metrics, TOPSIS scoring, and historical data."""

    try:
        # Load Step 6 (Candidates)
        df_candidates = pd.read_csv(os.path.join(job_dir, 'step6_suggested_candidates.csv'))
        
        # Load Step 5 (Historical Data) - Robust Load
        df_history = pd.DataFrame()
        try:
            files = [f for f in os.listdir(job_dir) if 'step5_ranked_samples' in f]
            if files:
                df_history = pd.read_csv(os.path.join(job_dir, files[0]))
        except:
            pass
        
        config = joblib.load(os.path.join(job_dir, 'config.json'))
        
        try:
            with open(os.path.join(job_dir, 'tea_config.json')) as f:
                tea_config = json.load(f)
        except:
            tea_config = None
    except FileNotFoundError:
        return {'error': 'Step 6 results not found. Please run generation first.'}
    
    if df_candidates.empty and df_history.empty:
        return {'error': 'No data available for analysis.'}
    
    # --- MERGE & SANITIZE DATASETS ---
    targets = config['targets']
    target_objs = config.get('objectives', {t: 'maximize' for t in targets})
    weights = config.get('weights', {t: 1.0 for t in targets})
    enable_cost = config.get('enable_cost', True)
    
    # Mark Source
    df_candidates['Source'] = 'AI Candidate'
    if not df_history.empty:
        df_history['Source'] = 'Historical Data'
        # Align History Columns to Prediction Columns
        for t in targets:
            actual_col = None
            if t in df_history.columns:
                actual_col = t
            else:
                for c in df_history.columns:
                    if c.strip() == t.strip():
                        actual_col = c
                        break
            
            if actual_col:
                df_history[f"{t}_pred"] = df_history[actual_col]
                df_history[f"{t}_uncertainty"] = 0.0  # History is ground truth
            else:
                df_history[f"{t}_pred"] = 0.0
                df_history[f"{t}_uncertainty"] = 0.0
        
        df = pd.concat([df_candidates, df_history], ignore_index=True)
    else:
        df = df_candidates.copy()
    
    # --- CRITICAL FIX: FORCE NUMERIC TYPES ---
    for t in targets:
        col = f"{t}_pred"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    cost_col = 'Total_Running_Cost'

    if cost_col in df.columns:
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
    else:
        df[cost_col] = 0
    
    # Performance Index
    perf_matrix = np.zeros(len(df))
    for t in targets:
        col_name = f"{t}_pred"
        if col_name not in df.columns:
            continue
        
        raw = df[col_name].values
        t_min, t_max = raw.min(), raw.max()
        
        if t_max > t_min:
            norm = (raw - t_min) / (t_max - t_min)
        else:
            norm = np.zeros_like(raw) + 0.5
        
        if target_objs.get(t) == 'minimize':
            norm = 1.0 - norm
        
        perf_matrix += norm * float(weights.get(t, 1.0))
    
    df['Performance_Index'] = perf_matrix
    
    # Cost efficiency (only when cost dimension is enabled, matching the rest of the analysis)
    if enable_cost:
        safe_cost = df[cost_col].replace(0, 1.0)
        df['Cost_Efficiency_Ratio'] = df['Performance_Index'] / safe_cost
    
    # Pareto frontier
    pareto_cols = [f"{t}_pred" for t in targets if f"{t}_pred" in df.columns]
    pareto_dirs = [target_objs.get(t, 'maximize') for t in targets if f"{t}_pred" in df.columns]
    
    if enable_cost and cost_col in df.columns:
        pareto_cols.append(cost_col)
        pareto_dirs.append('minimize')
    
    if pareto_cols:
        cost_matrix = []
        for col, direction in zip(pareto_cols, pareto_dirs):
            vals = df[col].values
            if direction == 'maximize':
                cost_matrix.append(-vals)
            else:
                cost_matrix.append(vals)
        
        cost_matrix = np.array(cost_matrix).T
        n_points = cost_matrix.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_pareto[i]:
                dominators = np.all(cost_matrix <= cost_matrix[i], axis=1) & \
                            np.any(cost_matrix < cost_matrix[i], axis=1)
                if np.any(dominators):
                    is_pareto[i] = False
        
        df['Is_Pareto_Optimal'] = is_pareto
    else:
        df['Is_Pareto_Optimal'] = False
    
    # --- TOPSIS RANKING ---
    topsis_cols = [f"{t}_pred" for t in targets if f"{t}_pred" in df.columns]
    topsis_objs = {f"{t}_pred": target_objs.get(t) for t in targets if f"{t}_pred" in df.columns}
    topsis_weights = {f"{t}_pred": float(weights.get(t, 1.0)) for t in targets if f"{t}_pred" in df.columns}
    
    if enable_cost:
        topsis_cols.append(cost_col)
        topsis_objs[cost_col] = 'minimize'
        avg_weight = sum(topsis_weights.values()) / (len(topsis_weights) if topsis_weights else 1)
        user_w = (tea_config or {}).get('cost_weight')
        try:
            user_w = float(user_w) if user_w not in (None, '') else None
        except (TypeError, ValueError):
            user_w = None
        # User-set cost weight (same scale as the response weights); default: mean response weight
        topsis_weights[cost_col] = user_w if (user_w is not None and user_w >= 0) else avg_weight
    
    df['TOPSIS_Score'] = calculate_topsis_score(df, topsis_cols, topsis_weights, topsis_objs)
    
    # Classification
    def classify(row):
        if row.get('Source') == 'Historical Data':
            return "Historical Reference"
        if row.get('Is_Pareto_Optimal') and row['TOPSIS_Score'] > 0.8:
            return "Elite (Pareto + Top Rank)"
        elif row.get('Is_Pareto_Optimal'):
            return "Pareto Optimal"
        elif enable_cost and 'Cost_Efficiency_Ratio' in row and row['Cost_Efficiency_Ratio'] > df['Cost_Efficiency_Ratio'].quantile(0.90):
            return "High Efficiency (Budget Pick)"
        elif row['TOPSIS_Score'] > 0.75:
            return "High Performer"
        else:
            return "Standard"
    
    df['Strategic_Category'] = df.apply(classify, axis=1)
    df.sort_values(['Is_Pareto_Optimal', 'TOPSIS_Score'], ascending=[False, False], inplace=True)
    
    # Add TEA breakdown if available
    if tea_config:
        tea_result = calculate_total_tea(tea_config, tea_config.get('scale', 1000))
        df['TEA_Materials_Pct'] = tea_result['cost_breakdown_percent']['materials']
        df['TEA_CapEx_Pct'] = tea_result['cost_breakdown_percent']['capex']
        df['TEA_OpEx_Pct'] = tea_result['cost_breakdown_percent']['opex']
    
    df.to_csv(os.path.join(job_dir, 'step7_pareto_analysis.csv'), index=False)
    
    return df
