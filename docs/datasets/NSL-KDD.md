# NSL-KDD

The implementation expects the two conventional NSL-KDD files:

- `KDDTrain+.txt`
- `KDDTest+.txt`

Store these files outside the repository and provide their paths at runtime. They must not be committed to Git.

## Provenance

The Canadian Institute for Cybersecurity dataset page states that NSL-KDD is no
longer distributed from UNB, while redistribution remains permitted. The local
copies were obtained from the public `Jehuty4949/NSL_KDD` mirror.

- Authoritative dataset description: https://www.unb.ca/cic/datasets/nsl.html
- Redistribution used for this reproduction: https://github.com/Jehuty4949/NSL_KDD

## Schema and labels

Each row contains the 41 KDD connection features, followed by the original
attack label and the NSL-KDD difficulty field. The loader removes `difficulty`
and maps labels into the five classes used by the base study: `Normal`, `DoS`,
`Probe`, `R2L`, and `U2R`.

The reproduction concatenates the published train and test files, removes exact
duplicates, and then creates a seeded stratified 70/30 split to match the split
reported in the base paper. This differs from some upstream scripts that use an
80/20 split; results should therefore always report the split explicitly.
