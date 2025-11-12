# src/features.py

#FEATURE ENGINEERING FUNCTIONS
# This file contains functions to extract, summarize, 
# and generate battle features for training models.

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils import safe_types

#Global dictionaries
# Dictionary with the status a Pokémon can have (to count the turns in which they occur)
STATUS_VALUES = {"slp": "slp", "frz": "frz", "par": "par"}
# Dictionary with the types of important moves
STALL_MOVES = {"recover","rest","softboiled","milkdrink","moonlight","morningsun","synthesis"}

# TYPE EFFECTIVENESS
def learn_type_effectiveness(battles, min_battles=15, alpha=1.5, beta=1.5, min_cap=0.75, max_cap=1.25):
    
    # We create two dictionaries to count results:
    # win[t1][t2]  -> how many times a Pokémon of type t1 has won against t2
    # cnt[t1][t2]  -> how many times a Pokémon of type t1 has faced t
    
    win = defaultdict(lambda: defaultdict(int))
    cnt = defaultdict(lambda: defaultdict(int))

    # We go through each battle in the dataset
    for b in battles:
        # We get player 1's team
        p1_team = b.get('p1_team_details') or []
        # We get player 2's initial Pokémon
        p2_lead = b.get('p2_lead_details') or {}
        # If any information is missing, we move on to the next battle
        if not p1_team or not p2_lead: 
            continue
        # We extract the types of the opponent's lead Pokémon (p2)
        p2_types = safe_types(p2_lead.get('types', []))
        # For each Pokémon in player 1's team
        for p1 in p1_team:
            # We extract that Pokémon's types ('fire', 'ghost'...)
            for t1 in safe_types(p1.get('types', [])):
                # And compare them with the opponent's lead Pokémon types 
                for t2 in p2_types:
                    # We count the matchup t1 vs t2
                    cnt[t1][t2] += 1
                    # If player 1 won the battle, we add a win to the pair (t1, t2)
                    if b.get('player_won') == 1:
                        win[t1][t2] += 1
                    # We also count the reverse matchup (t2 vs t1)
                    cnt[t2][t1] += 1
                    # If player 2 won, we add the win to the opponent's type
                    if b.get('player_won') == 0:
                        win[t2][t1] += 1
                        
    # We create the dictionary that will contain the final effectiveness values
    # eff[t1][t2] = "how effective type t1 is against type t2"
    # We initialize it with 1.0 (neutral) by default
    eff = defaultdict(lambda: defaultdict(lambda: 1.0))
    # We go through all the type pairs that exist in the data (t1 = attacker, t2 = opponent)
    for t1 in cnt:
        for t2 in cnt[t1]:
            n = cnt[t1][t2] # total number of matchups between t1 and t2
            
            # If the number of matchups is too low (by default < 15)
            # we don’t trust the result because it could be due to randomness.
            # In that case, we leave the effectiveness as neutral (1.0)
            if n < min_battles:
                eff[t1][t2] = 1.0
            else:
                '''
                We calculate the win rate "wr" of type t1 against t2.
                We add the parameters alpha and beta as smoothing factors:
                  - alpha acts as "fictional wins"
                  - beta acts as "fictional losses"
                This adjustment prevents extreme results (0% or 100%) when there are very few battles between those types.
                In practice, it corrects the statistics for rare types with limited data and improves the model’s stability.'''
                wr = (win[t1][t2] + alpha) / (n + alpha + beta) 
                
                '''We convert that win rate (wr) into an effectiveness value.
                    If wr = 0.5 → val = 1.0 (neutral)
                    If wr > 0.5 → val > 1.0 (advantage)
                    If wr < 0.5 → val < 1.0 (disadvantage)'''

                val = 0.5 + wr   

                ## We limit extreme values to prevent a type from appearing invincible or useless.
                # np.clip() trims values outside the range [0.75, 1.25].
                eff[t1][t2] = float(np.clip(val, min_cap, max_cap))
    return eff

def summarize_timeline(battle, max_turns=30):
    # We take the first turns of the battle
    tl = (battle.get('battle_timeline') or [])[:max_turns]
    # We define all the variables we want to return
    base = {
        'n_turns','p1_switches','p2_switches',
        'p1_status_count','p2_status_count',
        'p1_slp_turns','p2_slp_turns','p1_frz_turns','p2_frz_turns','p1_par_turns','p2_par_turns',
        'p1_ko_count','p2_ko_count',
        'p1_avg_hp_pct','p2_avg_hp_pct','p1_final_hp','p2_final_hp',
        'p1_damage_per_turn','p2_damage_per_turn',
        'lead_changes','hp_diff_mean','hp_diff_trend',
        'dominance_turns','final_lead_flag',
        'p1_stall_moves','p2_stall_moves'
    }
    # If the timeline is empty (battle without data), we return all zeros
    if not tl:
        return {k: 0.0 for k in base}

    p1_hp, p2_hp = [], []                 # lists with HP (%) turn by turn
    p1_switches = p2_switches = 0         # how many Pokémon switches each player made
    p1_status_total = p2_status_total = 0 # total turns with a status condition
    p1_slp = p2_slp = p1_frz = p2_frz = p1_par = p2_par = 0
    p1_ko = p2_ko = 0                     # how many Pokémon were knocked out (fainted)
    p1_stall = p2_stall = 0               # number of healing moves (stall)
    prev1 = prev2 = None                   # to detect Pokémon switches by the active Pokemon's name
    lead_flags = []                         # indicator of who is leading each turn



    # Dictionaries to store the last HP of each Pokémon on the field
    # Damage is only measured when the same Pokémon continues fighting
    last_hp1_by_mon = {}
    last_hp2_by_mon = {}
    total_dmg_on_p1 = 0.0
    total_dmg_on_p2 = 0.0

    # We go through the battle turn by turn
    for t in tl:
        # Status of the active Pokemon
        p1s, p2s = t.get('p1_pokemon_state'), t.get('p2_pokemon_state')
        # Details of the move used by player 1 and player 2 in that turn
        m1, m2 = t.get('p1_move_details'), t.get('p2_move_details')

        if p1s:
            name1 = p1s.get('name')
            # Current HP percentage of the active Pokemon, 0 if there is no value
            hp1 = float(p1s.get('hp_pct', 0.0))
            # If the active Pokémon's name changes, increment p1_switches by 1
            if prev1 and name1 != prev1: p1_switches += 1
            prev1 = name1
            # We store the Pokémon's HP in a list
            p1_hp.append(hp1)
            st = p1s.get('status')
            if st and st != 'nostatus':
                # If it has a status condition, increment the counter by 1
                p1_status_total += 1
                if st == 'slp': p1_slp += 1  #Sleep
                elif st == 'frz': p1_frz += 1 #Freeze
                elif st == 'par': p1_par += 1 #Paralysis
                elif st == 'fnt': p1_ko += 1 #Fainted/KO

            # If this Pokémon (name1) appears for the first time, we store its current HP as the initial reference point.
            if name1 not in last_hp1_by_mon:
                last_hp1_by_mon[name1] = hp1
            else:
                # If the Pokémon was already in battle, we calculate the change in its HP since the previous turn 
                delta = last_hp1_by_mon[name1] - hp1
                # If it’s positive, it’s damage and therefore we add it to the counter
                if delta > 0: 
                    total_dmg_on_p1 += float(delta)
                # We update the Pokémon's HP for the next turn
                last_hp1_by_mon[name1] = hp1
                
            # We check if the move used by player 1 (m1) belongs to the list of defensive or healing moves (STALL_MOVES).
            # If so, we increment the counter p1_stall. This helps detect more defensive playstyles ("stall teams")
            if m1 and (m1.get('name','').lower() in STALL_MOVES):
                p1_stall += 1
                
        #The same for player 2 
        if p2s:
            name2 = p2s.get('name')
            hp2 = float(p2s.get('hp_pct', 0.0))
            if prev2 and name2 != prev2: p2_switches += 1
            prev2 = name2
            p2_hp.append(hp2)

            st2 = p2s.get('status')
            if st2 and st2 != 'nostatus':
                p2_status_total += 1
                if st2 == 'slp': p2_slp += 1
                elif st2 == 'frz': p2_frz += 1
                elif st2 == 'par': p2_par += 1
                elif st2 == 'fnt': p2_ko += 1

            if name2 not in last_hp2_by_mon:
                last_hp2_by_mon[name2] = hp2
            else:
                delta = last_hp2_by_mon[name2] - hp2
                if delta > 0: total_dmg_on_p2 += float(delta)
                last_hp2_by_mon[name2] = hp2

            if m2 and (m2.get('name','').lower() in STALL_MOVES):
                p2_stall += 1
                
        # Who is leading this turn (True if P1 has higher HP%)
        if p1s and p2s:
            lead_flags.append(p1s.get('hp_pct',0.0) > p2s.get('hp_pct',0.0))
            
    # Returns the average damage received per turn, normalized by the battle duration
    def mean_per_turn(total_damage, turns):
        return float(total_damage / max(1, turns))

    # Measures the battle’s instability by checking if there are many changes in who is leading across turns
    if len(lead_flags) > 1:
        lead_changes = int(np.sum(np.array(lead_flags[1:], dtype=bool) != np.array(lead_flags[:-1], dtype=bool)))
    else:
        lead_changes = 0

    if p1_hp and p2_hp:
        hp_diff = np.array(p1_hp, dtype=float) - np.array(p2_hp, dtype=float) # HP% difference between both players in each turn
        hp_diff_mean = float(hp_diff.mean()) # Average of those HP% differences
        x = np.arange(len(hp_diff), dtype=float)
        
        # The variable 'slope' measures the trend of the HP difference between both players.
        # If the slope is positive, it means P1 is improving or making a comeback during the battle.
        # If negative, P1 is progressively losing.
        # If close to 0, the battle remains balanced in terms of HP.
        slope = float(np.polyfit(x, hp_diff, 1)[0]) if len(hp_diff) >= 2 else 0.0
        dominance_turns = float((hp_diff > 0).sum()) #Number of turns P1 was ahead
        final_lead_flag = 1.0 if hp_diff[-1] > 0 else 0.0 # 1 if P1 ends with more HP than P2
        # Final HP of each side
        p1_final_hp = float(p1_hp[-1])
        p2_final_hp = float(p2_hp[-1])

    # If any HP is empty, return 0
    else:
        hp_diff_mean = slope = dominance_turns = final_lead_flag = 0.0
        p1_final_hp = p2_final_hp = 0.0

    return {
        'n_turns': float(len(tl)),
        'p1_switches': float(p1_switches), 'p2_switches': float(p2_switches),
        'p1_status_count': float(p1_status_total), 'p2_status_count': float(p2_status_total),
        'p1_slp_turns': float(p1_slp), 'p2_slp_turns': float(p2_slp),
        'p1_frz_turns': float(p1_frz), 'p2_frz_turns': float(p2_frz),
        'p1_par_turns': float(p1_par), 'p2_par_turns': float(p2_par),
        'p1_ko_count': float(p1_ko), 'p2_ko_count': float(p2_ko),
        'p1_avg_hp_pct': float(np.mean(p1_hp) if p1_hp else 0.0),
        'p2_avg_hp_pct': float(np.mean(p2_hp) if p2_hp else 0.0),
        'p1_final_hp': p1_final_hp, 'p2_final_hp': p2_final_hp,
        'p1_damage_per_turn': mean_per_turn(total_dmg_on_p1, len(tl)),
        'p2_damage_per_turn': mean_per_turn(total_dmg_on_p2, len(tl)),
        'lead_changes': float(lead_changes), 'hp_diff_mean': float(hp_diff_mean),
        'hp_diff_trend': float(slope), 'dominance_turns': float(dominance_turns),
        'final_lead_flag': float(final_lead_flag),
        'p1_stall_moves': float(p1_stall), 'p2_stall_moves': float(p2_stall),
    }

def team_static_features(p_team, prefix='p1'):
    # Dictionary where we will store all the generated features
    out = {}
    # List of the six base stats of a Pokémon
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    if not p_team:
        # If the team has no data, fill all columns with 0
        for s in stats:
            for suf in ['mean','max','min','std']:
                out[f'{prefix}_{s}_{suf}'] = 0.0
        out[f'{prefix}_type_count'] = 0.0
        out[f'{prefix}_type_entropy'] = 0.0
        return out

    # We create a dictionary with the numerical values of each base stat (HP, Atk, Def, etc.)
    # for all Pokémon on the team. If any stat is missing, we use 0.0.
    stat_vals = {s:[float(p.get(s,0.0)) for p in p_team] for s in stats}
    # Calculate the average HP of the entire team, the Pokémon with the highest and lowest HP,
    # and the variation between them
    for s in stats:
        vals = np.array(stat_vals[s], dtype=float)
        out[f'{prefix}_{s}_mean'] = float(vals.mean())
        out[f'{prefix}_{s}_max']  = float(vals.max())
        out[f'{prefix}_{s}_min']  = float(vals.min())
        out[f'{prefix}_{s}_std']  = float(vals.std())

    # Create a list with all the types of all Pokémon on the team
    types_flat = []
    for p in p_team:
        types_flat += [t for t in safe_types(p.get('types', [])) if t != 'notype']
    unique, counts = np.unique(types_flat, return_counts=True)
    # Number of distinct types in the team
    out[f'{prefix}_type_count'] = float(len(unique))
    # Calculate how much diversity there is among Pokémon in the team
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([1.0])
    out[f'{prefix}_type_entropy'] = float(-(probs * np.log(probs + 1e-12)).sum())
    return out

def create_features_from_battle(b, type_eff):
    '''
    Extracts and combines all relevant features from a Pokémon battle.
    Calculates player team statistics, opponent lead data, type advantages, attribute differences, 
    and a summary of the battle (damage, turns, status conditions, control).
    Returns a dictionary with all these variables ready to train the victory prediction model.
    '''
    # Dictionary where all features are stored
    f = {}
    # List with all Pokémon of player 1
    p1_team = b.get('p1_team_details') or []
    # Details of the opponent's lead Pokémon
    p2_lead = b.get('p2_lead_details') or {}

    # Add static features of player 1's team
    f.update(team_static_features(p1_team, 'p1'))

    # # Extract the base stats of player 2's lead Pokémon 
    for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']:
        f[f'p2_lead_{s}'] = float(p2_lead.get(s,0.0)) if p2_lead else 0.0
    #And also it type
    p2_types = safe_types(p2_lead.get('types', [])) if p2_lead else ['notype']

    # Calculate how much advantage or disadvantage player 1's team has against the opponent's lead type.
    p1_types = [t for p in p1_team for t in safe_types(p.get('types', [])) if t!='notype']
    if p1_types and p2_types:
        #effectiveness of P1 → P2 (how effective P1's attacks are)
        vals = [type_eff[t1][t2] for t1 in p1_types for t2 in p2_types]
        #effectiveness of P2 → P1 (how vulnerable P1 is)
        rvs  = [type_eff[t2][t1] for t1 in p1_types for t2 in p2_types]
        f['type_eff_mean'] = float(np.mean(vals))
        f['type_eff_max']  = float(np.max(vals))
        f['type_eff_min']  = float(np.min(vals))
        f['type_advantage'] = f['type_eff_mean'] - 1.0
        f['type_disadvantage'] = float(np.mean(rvs)) - 1.0
    else:
        f['type_eff_mean'] = 1.0
        f['type_eff_max'] = 1.0
        f['type_eff_min'] = 1.0
        f['type_advantage'] = 0.0
        f['type_disadvantage'] = 0.0

    # Calculate the base stat advantage of player 1 against the opponent's lead Pokémon
    for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']:
        f[f'diff_{s}'] = f.get(f'p1_{s}_mean',0.0) - f.get(f'p2_lead_{s}',0.0)

    f['total_stat_advantage'] = sum(f.get(f'p1_{s}_mean',0.0) for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']) - \
                                sum(f.get(f'p2_lead_{s}',0.0) for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe'])

    # Add features from the battle timeline
    f.update(summarize_timeline(b))

    # Derived features:
    # Balanced damage: who has received more damage per turn
    # Average HP ratio: which player maintains more HP
    # Leadership volatility: how many times the lead changes
    # Status difference: how many status conditions each player had

    f['damage_balance'] = f['p2_damage_per_turn'] - f['p1_damage_per_turn']  # ojo: “daño recibido”, invertir si prefieres
    f['hp_ratio'] = (f['p1_avg_hp_pct']+1e-3) / (f['p2_avg_hp_pct']+1e-3)
    f['lead_volatility'] = f['lead_changes'] / max(1.0, f['n_turns'])
    f['status_diff'] = (f['p1_status_count'] - f['p2_status_count'])

    # Combine variables to capture complex interactions:
    # type_adv_x_total: combines type and stat advantage
    # stall_diff: who used more healing moves
    # *_adv: advantages for each status type (paralysis, sleep, freeze, etc.)
    # ko_adv: if the opponent has knocked out more Pokémon

    f['type_adv_x_total'] = f['type_advantage'] * f['total_stat_advantage']
    f['status_x_turns'] = f['status_diff'] / max(1.0, f['n_turns'])
    f['stall_diff'] = f['p1_stall_moves'] - f['p2_stall_moves']
    f['freeze_adv'] = f['p1_frz_turns'] - f['p2_frz_turns']
    f['sleep_adv']  = f['p1_slp_turns'] - f['p2_slp_turns']
    f['para_adv']   = f['p1_par_turns'] - f['p2_par_turns']
    f['ko_adv']     = f['p2_ko_count'] - f['p1_ko_count']  # si P2 tiene más KOs, P1 en peor situación

    # Add the battle_id (battle number)
    f['battle_id'] = b.get('battle_id', -1)
    # Add the label (player_won: 1 if the player won, 0 if lost)
    if 'player_won' in b:
        f['player_won'] = int(b['player_won'])
    # Return the final dictionary with all features
    return f

def build_feature_df(raw_list, type_eff):
    '''
    Iterate over each battle and apply the function create_features_from_battle() to each one
    '''
    feats = [create_features_from_battle(b, type_eff) for b in tqdm(raw_list, desc='Features')]
    return pd.DataFrame(feats).fillna(0.0)