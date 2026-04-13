import pandas as pd
ml_docs = pd.read_csv('data/processed/mathlib_docstrings.csv')

checks = [
    'Mathlib.Algebra.Field.ZMod',
    'Mathlib.Data.ZMod',
    'Mathlib.Data.List.Prime',
    'Mathlib.Data.Nat.Prime',
    'Mathlib.NumberTheory.Cyclotomic.Rat',
    'Mathlib.Data.Rat',
    'Mathlib.RingTheory.Int.Basic',
    'Mathlib.Data.Int',
    'Mathlib.GroupTheory.GroupAction.Quotient',
    'Mathlib.GroupTheory.QuotientGroup',
    'Mathlib.RingTheory.LocalRing.Quotient',
    'Mathlib.RingTheory.WittVector.FrobeniusFractionField',
    'Mathlib.GroupTheory.FreeAbelianGroup',
    'Mathlib.GroupTheory.Coset.Basic',
    'Mathlib.GroupTheory.IsSubnormal',
    'Mathlib.Data.Nat.Digits.Div',
    'Mathlib.Data.Finite.Set',
    'Mathlib.Algebra.MonoidAlgebra.Division',
]
for name in checks:
    row = ml_docs[ml_docs['module_name']==name]
    short = name.replace('Mathlib.','')
    if not row.empty:
        doc = str(row.iloc[0]['docstring'])[:110].replace('\n',' ')
        print(short[:44].ljust(45), '|', doc)
    else:
        print(short[:44].ljust(45), '| (not in docs)')
