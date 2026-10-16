# Turing Bohr Molecular Platform

A specification-first, test-driven research platform for tracking molecular
scientific objects — their acquisition provenance, their identity, and the
transformations between them — through an auditable filesystem record. Each
experiment keeps an append-only activity log; most scientific registrations
(inputs, entities, states, conformers) are create-once and are never
overwritten once persisted — but a record that tracks an in-progress
operation's own status (an execution or promotion *attempt*) is the
deliberate exception: it is updated in place as that operation proceeds. See
[Scientific and architectural limits](#scientific-and-architectural-limits)
for the exact boundary. It is an early-stage internal research tool, not a
finished product, and not a substitute for domain expertise in any step it
touches.

This README describes **only what is implemented and tested in this
repository today**. It does not describe a roadmap or a vision statement.

## What this platform is for

Each unit of work lives inside a self-contained **experiment** directory,
created by `init` and containing its own inputs, scientific records, and
append-only activity log. Everything below operates on one experiment at a
time; nothing is shared implicitly across experiments.

## What is implemented today

- **Ligand and protein input registration** (`ligand add`, `protein add`,
  `ligand show`, `protein show`): register a RAW file you already have
  (uploaded) or one downloaded from a named external source, with full
  acquisition provenance (method, source, checksum).
- **PubChem acquisition** (`ligand fetch-pubchem`): download a compound's 2D
  SDF structure directly from the public
  [PubChem PUG REST API](https://pubchem.ncbi.nlm.nih.gov/rest/pug/) by CID,
  and register it as a ligand input with full provenance back to the
  originating CID and URL. This is a network download of an already-existing
  public record — never a lookup against an internal database, and never a
  claim about what PubChem's own website shows you when you view that CID
  there.
- **Scientific identity** (`entity add/show`, `state add/show/list`,
  `conformer add/show/list`): register a `ScientificEntity` (the stable
  identity of a molecular object across an experiment), a `MolecularState`
  (a specific, explicitly identified chemical realization of that entity —
  e.g. a protonation or tautomer state), and a `Conformer` (one specific,
  immutable geometry of one state). See
  [Concepts](docs/tutorial-pubchem-caffeine.md#concepts-before-the-first-command)
  in the tutorial for the exact, non-overlapping boundary between these and
  a PubChem compound record / a RAW file.
- **Ligand promotion** (`ligand promote`): the tested, audited path from a
  registered ligand RAW artifact to its first `ScientificEntity` +
  `MolecularState` pair. This is a coordinated, multi-step operation — not
  a single indivisible transaction across every file it touches: it
  snapshots the RAW, inspects it with RDKit (pinned version), reserves
  identifiers, creates-or-reuses the entity, registers the state, and only
  then publishes the promotion record. If a later step fails after an
  earlier one has already written something, best-effort compensation
  reverses what that same attempt created; once the promotion record
  itself is committed, a failure afterwards never removes that record,
  the entity, or the state it just created — compensation only ever
  applies before publication, never after. It verifies the RAW checksum
  at the moment of promotion and reports the exact engine version and any
  structural warnings it raised.
- **State transformation provenance** (`state derive`): records that a
  already-computed derived state (e.g. a protonation or a tautomer state you
  produced some other way) came from a named parent state through a named
  operation. **It does not perform the chemistry itself** — it is a
  provenance ledger entry for a transformation you already did, never a
  protonation or tautomer assignment engine.
- **Tautomer enumeration** (`state enumerate-tautomers`): a separate,
  RDKit-backed capability that enumerates candidate tautomer structures for
  an existing SDF-represented state and publishes each as its own new state
  with full provenance. It enumerates candidates mechanically; it does not
  decide which one, if any, is the biologically or energetically "correct"
  one. Not covered by the tutorial in this repository — see its own CLI help
  (`molecular state enumerate-tautomers --help`) and the Increment 016/017
  specifications for its exact contract.
- **Experimental conditions declarations** (`conditions add/revise/show/list`):
  record a scientist's own declared pH conditions for an experiment, as an
  immutable, create-once record — never inferred, defaulted, or rewritten; a
  revision is always a new record, never an edit of the one it supersedes.
- **Receptor preparation foundation** (`receptor ...`, `protein inspect`):
  inspection of a receptor structure file (mmCIF/legacy PDB via Gemmi),
  explicit chain/assembly/water/metal/component **selection** declarations,
  and their deterministic **resolution** against the inspected structure.
  "Read-only" here describes the original structure file specifically — it
  is never modified, and no step in this pipeline ever materializes a new,
  prepared coordinate file from it. The pipeline itself does persist its
  own records (inspection reports, declared selection policies,
  resolutions) — inspecting, selecting, and resolving are not no-op reads.
  This stage is strictly logical: it never adds hydrogens, never assigns
  protonation states or partial charges, never performs docking, and never
  classifies a component by biological function — only by mechanical,
  format-declared signals (water / metal-ion / modified-polymer-residue /
  other-non-polymer / unknown). See `molecular receptor --help`.

## What is explicitly not implemented

- Chemical preparation of any kind: protonation assignment, pKa calculation,
  partial charges, geometry minimization, solvation, membrane insertion.
- Conformer *generation*: `conformer add` records a geometry you already
  have; nothing in this platform computes, optimizes, or validates one.
- Docking, scoring, pose ranking, or any search-space definition.
- Structural materialization of a receptor selection into a new coordinate
  file (`.pdb`/`.mmcif`).
- Any AI-driven or automatic scientific decision (choosing a tautomer, a
  protonation state, a "best" conformer, a binding site).

## Installation (clean environment)

Requires Python ≥ 3.11.

```powershell
git clone https://github.com/sysoutemanuelhenrique/molecular-platform.git
cd molecular-platform
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[rdkit,structure,dev]"
```

- `rdkit` is required for `ligand promote` and `state enumerate-tautomers`.
- `structure` (Gemmi) is required for every `receptor`/`protein inspect`
  command.
- `dev` is only needed to run this repository's own test suite (`pytest`).

Installing the package exposes two equivalent console entry points,
`molecular` and `tbohr` — both run the exact same CLI.

```powershell
molecular --help
```

These install steps themselves were not re-executed as part of preparing
the tutorial below — the tutorial's commands were validated against this
repository's already-existing `.venv` (with `rdkit`/`gemmi` already
installed), not a fresh clone. See the tutorial's own
[Prerequisites](docs/tutorial-pubchem-caffeine.md#prerequisites) section.

## Tutorial

The main, reproducible tutorial — registering caffeine (PubChem CID 2519)
end to end, with every command's exact output — lives at
[`docs/tutorial-pubchem-caffeine.md`](docs/tutorial-pubchem-caffeine.md). It
also includes a short alternative-examples section (rosmarinic acid,
imidacloprid, diclofenac, atrazine).

## Scientific and architectural limits

This platform makes the following specific guarantees — deliberately not
phrased as one universal rule, because the actual contract differs by
command:

- Every RAW artifact is checksummed and never silently overwritten or
  mutated in place.
- The basic registration commands — `ligand add`, `protein add`,
  `entity add`, `state add`, `conformer add` — are create-once in the
  strict sense: a colliding id (exact or casefold) is always rejected
  outright, even when the new content would have been identical to what
  is already there. These commands never recognize a retry as a reuse; a
  second call under the same id always fails.
- A small number of higher-level services implement idempotent reuse on
  top of that, specifically: `conditions add`/`conditions revise`,
  `ligand promote`, and `protein inspect`/`receptor select`/
  `receptor resolve`. Each of these compares a retry's content against
  the already-persisted record and reports `reused` instead of failing
  when the content is equivalent — rejecting only a genuine divergence
  under the same id. This reuse behavior is specific to these services; it
  is not a property of the basic registration commands above.
- The receptor inspection pipeline distinguishes identity from measurement
  when the source structure file omits a field — it does not make one
  universal "absence never becomes a default" guarantee. An *identity*
  field it cannot determine (`label_asym_id`, `label_entity_id`, and
  similarly) is always `None`, never guessed. Some other, non-identity
  fields do use an explicit, documented default when the source omits
  them — for example, an atom's `occupancy` defaults to `1` when the
  mmCIF file does not declare one
  (`src/molecular/receptor_parsing.py`). Which rule applies to which
  field is defined field by field in the receptor specification itself,
  not reproduced here in full.
- No step in this platform infers, recommends, or automatically selects a
  scientific choice (a tautomer, a protonation state, a chain, a docking
  pose) on the scientist's behalf. Every scientific decision is an explicit,
  attributed declaration from a human. When exactly one option is actually
  available — for example, a receptor inspection report naming exactly one
  model — resolving it without a separate explicit choice is a mechanical
  consequence of there being no ambiguity to resolve, never a judgment
  about which option is scientifically preferable. The moment more than
  one option exists, an explicit declaration is always required.
- The create-once/idempotent-reuse guarantees above apply to finished
  scientific records. They do **not** apply to an attempt's own status
  record (`ExecutionAttempt`, a ligand promotion attempt): that record is
  explicitly rewritten as the attempt progresses from `started` to
  `resolved` to a terminal outcome. Once an attempt reaches a terminal
  status it is never reopened — a retry is always a brand new attempt —
  but the in-progress record itself is mutable by design, not an exception
  to be confused with the guarantees above.

## License

No license file is currently present in this repository. Until one is
added, no rights beyond viewing this source are granted or implied.
