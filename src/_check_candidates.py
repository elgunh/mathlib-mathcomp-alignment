"""Check top-10 candidates for hard modules to understand hierarchical opportunities."""
import pandas as pd

df = pd.read_csv('outputs/iterative_matches_v3_textv3.csv')

hard = ['bigop', 'fingraph', 'finset', 'fraction', 'eqtype', 'abelian',
        'automorphism', 'morphism', 'div', 'intdiv', 'gseries', 'path',
        'prime', 'zmodp', 'quotient', 'ring_quotient', 'rat', 'ssrint']

gold_prefixes = {
    'bigop': 'Mathlib.Algebra.BigOperators',
    'fingraph': 'Mathlib.Combinatorics.SimpleGraph',
    'finset': 'Mathlib.Data.Finset',
    'fraction': 'Mathlib.RingTheory.Localization',
    'eqtype': 'Mathlib.Logic.Equiv',
    'abelian': 'Mathlib.GroupTheory.Abelian',
    'automorphism': 'Mathlib.GroupTheory.Aut',
    'morphism': 'Mathlib.GroupTheory.GroupHom',
    'div': 'Mathlib.Data.Nat.Div',
    'intdiv': 'Mathlib.Data.Int',
    'gseries': 'Mathlib.GroupTheory.Series',
    'path': 'Mathlib.Combinatorics.SimpleGraph',
    'prime': 'Mathlib.Data.Nat.Prime',
    'zmodp': 'Mathlib.Data.ZMod',
    'quotient': 'Mathlib.GroupTheory.QuotientGroup',
    'ring_quotient': 'Mathlib.RingTheory.Ideal.Quotient',
    'rat': 'Mathlib.Data.Rat',
    'ssrint': 'Mathlib.Data.Int',
}

for mc in hard:
    sub = df[df['mathcomp_module']==mc].sort_values('final_score', ascending=False)
    if sub.empty:
        print(f'{mc}: NO CANDIDATES')
        continue
    gold = gold_prefixes.get(mc, '')
    print(f'\n{mc} (gold prefix: {gold})')
    for i, (_, row) in enumerate(sub.head(10).iterrows()):
        ml = row['mathlib_module']
        sc = row['final_score']
        hit = ml.startswith(gold) if gold else False
        mark = '<<GOLD' if hit else ''
        print(f'  {i+1}. {ml:55s} {sc:.3f} {mark}')
