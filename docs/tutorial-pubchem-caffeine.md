# Tutorial: Registering Caffeine from PubChem

This tutorial walks through this platform's ligand acquisition and
promotion commands — acquire a compound from PubChem, register it, and
promote it into a scientific identity — using **caffeine (PubChem CID
2519)** as the running example. Every command and every line of output
below was executed against this repository's own code, in a disposable
experiment directory outside the repository, using this repository's
**already-existing** virtual environment (see
[Prerequisites](#prerequisites) for exactly what that does and does not
validate), and is reproduced exactly as observed (with run-specific values
annotated as such — see
[Values that change on every run](#values-that-change-on-every-run)).

This is not a chemistry tutorial. It demonstrates what this specific
software does and does not do — nothing here should be read as guidance
on caffeine's pharmacology, safety, or any other scientific property of the
molecule itself.

## Concepts before the first command

This platform tracks five distinct things in this tutorial, and conflating
any two of them is the single most common source of confusion:

| Concept | What it actually is | Where it lives |
|---|---|---|
| **PubChem Compound** | A public record on the NIH PubChem website, identified by a CID (e.g. CID 2519). You can view it externally at `https://pubchem.ncbi.nlm.nih.gov/compound/2519` — in your browser, outside this platform. This tutorial never renders or validates that page; it only downloads one specific file format from it. | PubChem's own servers — not this repository. |
| **RAW file** | The exact bytes this platform downloaded from PubChem (a 2D SDF) or that you uploaded yourself, checksummed and stored unmodified under the experiment's `inputs/` directory. A RAW file is acquisition provenance, not a scientific claim. | `inputs/ligands/<id>/` inside your experiment. |
| **ScientificEntity** | The stable scientific identity of "this molecular object" within the experiment — independent of file format, state, or external id. It has exactly three fields: `schema_version`, `id`, and `name` — nothing else, and no PubChem CID of its own. | `scientific/entities/<id>.yaml` |
| **MolecularState** | One explicitly identified chemical realization of a `ScientificEntity` (e.g. "the initial state as parsed from the RAW file"), referencing one stored representation file. It is **not** a conformer, a pose, or a file format — it is an identity. | `scientific/states/<id>.yaml` + its representation artifact |
| **Conformer** | One specific, immutable 3D geometry of exactly one `MolecularState`. Registering a Conformer **never asserts that the geometry is genuinely three-dimensional or chemically valid** — this platform has no conformer-generation or validation engine. See [Conformers in this platform](#conformers-in-this-platform) below for exactly what that means in practice, and why this tutorial does not register one. | `scientific/conformers/<id>.yaml` + its representation artifact |

Only the **RAW file**, **ScientificEntity**, **MolecularState**, and
**Conformer** are things this software actually creates, verifies, or
stores. The **PubChem Compound** page is something you may look at in a
browser for your own understanding; this platform never scrapes it, never
renders it, and never asserts anything about what it shows.

## Prerequisites

Follow [Installation](../README.md#installation-clean-environment) in the
main README first: a Python ≥ 3.11 virtual environment with
`pip install -e ".[rdkit,structure,dev]"`, and the `molecular` (or `tbohr`)
command on your `PATH`. You also need outbound network access to
`pubchem.ncbi.nlm.nih.gov` on port 443.

**What was actually validated.** This tutorial's commands were run in this
repository's **pre-existing** `.venv`, already built beforehand with
`rdkit`/`gemmi` installed. This validation did **not** start from a fresh
clone and did **not** re-execute the clean-environment install sequence
above — that sequence is the standard editable install for this project,
but it was not itself re-verified while preparing this tutorial. If
anything in `pip install -e ".[rdkit,structure,dev]"` itself fails for you,
that is not something this tutorial has confirmed works.

Run every command below from **outside this repository**, in a directory
you are using only for this tutorial. On Windows, keep that directory's
path short — this platform stages artifacts under nested subdirectories,
and a very deeply nested starting path can exceed Windows' traditional
260-character path limit during `ligand promote` (observed directly while
preparing this tutorial). `C:\Users\<you>\tbohr-demo\` is a safe choice;
a path buried many directories deep under a temp folder is not.

## 1. Create an experiment

```powershell
molecular init caffeine-demo --title "Caffeine PubChem Demo" --responsible "Your Name" --summary "Academic demonstration"
```

Observed output:

```
Experiment 'caffeine-demo' created successfully.
```

This creates `caffeine-demo/` in your current directory, with
`experiment.yaml`, `inputs/`, `artifacts/`, `results/`, and `logs/` inside
it. Every command below names this same experiment (`caffeine-demo`)
explicitly — nothing is implicit or "current."

## 2. Acquire caffeine from PubChem (CID 2519)

```powershell
molecular ligand fetch-pubchem caffeine-demo caffeine-pubchem --cid 2519 --name "Caffeine"
```

Observed output:

```
Ligand 'caffeine-pubchem' registered successfully.
```

`caffeine-pubchem` is the **ligand input id** you chose for this command —
you will reuse it, verbatim, in step 4. This step performed one real HTTP
request to
`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2519/SDF?record_type=2d`
and stored the returned bytes as this ligand's RAW artifact. No chemistry
engine was invoked here — this step is acquisition and registration only.

## 3. Inspect what was registered

```powershell
molecular ligand show caffeine-demo caffeine-pubchem
```

Observed output (your `Acquired at` timestamp will differ, and your
`Checksum` may differ — see
[below](#values-that-change-on-every-run)):

```
LIGAND

ID: caffeine-pubchem
Name: Caffeine
External ID: CID 2519
Acquisition method: downloaded
Source name: PubChem
Source identifier: CID 2519
Source URI: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2519/SDF?record_type=2d
Acquired at: 2026-10-14T20:13:28.681427+00:00

Raw artifact:
Stored path: inputs/ligands/caffeine-pubchem/pubchem_cid_2519_2d.sdf
Resolved source path: <your absolute path>\caffeine-demo\inputs\ligands\caffeine-pubchem\pubchem_cid_2519_2d.sdf
Format: sdf
Checksum: 18cd70c23fa2606e08159af3b3815753d8f6080da5ea651a2403fb370ce43213
```

This is the full acquisition record: what was asked for (CID 2519), where
it came from (the exact URL), and the SHA-256 of exactly what is stored on
disk. Nothing here has been interpreted as chemistry yet.

## 4. Promote the ligand into a scientific identity

```powershell
molecular ligand promote caffeine-demo caffeine-pubchem --entity caffeine --state caffeine-initial --name "Caffeine"
```

Observed output (your `Attempt ID` will differ; your `Promotion ID` and
checksum may differ — see
[Values that change on every run](#values-that-change-on-every-run)):

```
LIGAND PROMOTION

Attempt ID: c4c24d2b0b264d15a6723a9b3aa4a40a
Promotion ID: 35f54a9c3cad194324168347da636f3e98360f692c719ab2f238fd37d29981f8
Input: caffeine-pubchem
Entity: caffeine (created)
Initial state: caffeine-initial (created)
Representation: scientific/artifacts/states/caffeine-initial/snapshot.sdf
RAW checksum verified: 18cd70c23fa2606e08159af3b3815753d8f6080da5ea651a2403fb370ce43213
Engine: rdkit 2026.03.6
```

`--entity caffeine` and `--state caffeine-initial` are ids **you choose**
here — they are what you will use in steps 5 and 6 below, not anything
printed by a previous step. This single command runs a coordinated,
multi-step operation, not one indivisible transaction across every file
it touches: it (1) re-verifies the RAW checksum against what step 2
actually stored, (2) parses and sanitizes the structure with the pinned
RDKit engine shown in `Engine:`, (3) creates or reuses the
`ScientificEntity`, and (4) registers the `MolecularState`, with its
representation stored at the `Representation:` path shown, before finally
publishing the promotion record itself. If a later one of these steps
fails after an earlier one has already written something, best-effort
compensation reverses what *this same attempt* created — but this is not
an infallible rollback guarantee, only a best effort. Once the promotion
record is itself published, a failure afterward never removes that
record, the entity, or the state — see the main
[README](../README.md#what-is-implemented-today) for the exact boundary.
`(created)` next to `Entity`/`Initial state` is this platform's own factual
record of what happened — never your assumption; running this exact command
a second time with the same arguments reports `reused` instead, not an
error.

If `ligand promote` reports `STORAGE ERROR` / `the staging file could not
be created: FileNotFoundError`: a too-long path for Windows was the cause
observed once while preparing this tutorial (see the note in
[Prerequisites](#prerequisites)) — check your path length first, since
it's the easiest thing to rule out. `FileNotFoundError` is a generic
operating-system signal, though, and can have other causes (a missing
intermediate directory, a permissions problem, a disconnected or
unavailable drive); do not assume path length is the cause if shortening
the path does not actually resolve it.

## 5. Show the registered entity

```powershell
molecular entity show caffeine-demo caffeine
```

Observed output:

```
ENTITY

Schema version: 1.0
ID: caffeine
Name: Caffeine
```

This is deliberately minimal: a `ScientificEntity` carries exactly three
fields — `schema_version`, `id`, and `name` — nothing else. In particular,
it never carries the PubChem CID; the CID and the download URL are
acquisition provenance of the **ligand input** from step 3, full stop.
What the state (and, if one exists, a conformer) carries instead is a
**representation reference** — a stored file's path, format, and
checksum, as shown in step 6 below — never the CID itself, and never a
copy of the acquisition record. A specific 3D geometry, if one existed,
would be a `Conformer`'s own concern, distinct again from both of the
above — see
[Conformers in this platform](#conformers-in-this-platform) further
below.

## 6. Show the registered state

```powershell
molecular state show caffeine-demo caffeine-initial
```

Observed output (checksum matches step 4's `RAW checksum verified`):

```
STATE

ID: caffeine-initial
Entity: caffeine
Representation path: scientific/artifacts/states/caffeine-initial/snapshot.sdf
Representation format: sdf
Representation checksum: 18cd70c23fa2606e08159af3b3815753d8f6080da5ea651a2403fb370ce43213
```

## Conformers in this platform

This tutorial stops at the `MolecularState` registered in step 6 above,
and **deliberately does not register a `Conformer`**.

The platform does provide a `conformer add` command, so the registration
mechanism itself is real and tested — but its own `--help` text says,
verbatim: *"This command does not generate a conformer - it records a
geometry the scientist already has. Conformer generation is not
implemented in this increment."* Using it meaningfully requires a genuine
3D geometry you already have from somewhere else. Nothing in this tutorial
produces one: the PubChem download in step 2 was explicitly the **2D** SDF
(`record_type=2d` — PubChem's own parameter), and no step between there and
here computes a 3D geometry. Registering that same 2D file again under the
`Conformer` identity would not demonstrate a real conformer — it would
simulate one, and this tutorial does not do that.

If and when you have a genuine 3D geometry for this state (produced by a
separate tool outside this platform today), the command is:

```powershell
molecular conformer add <experiment> <conformer_id> --state <state_id> --source-file <path-to-your-3d-file> --format <format>
```

See `molecular conformer add --help` and `molecular conformer show --help`
for the full, exact contract.

## What this tutorial deliberately did not do

- It did not register a `Conformer` — see
  [Conformers in this platform](#conformers-in-this-platform) above for
  why: doing so here would have required either a real 3D geometry (which
  nothing in this tutorial produces) or a simulated one, which this
  tutorial does not do.
- It did not add hydrogens, assign a protonation state, or compute a pKa.
- It did not enumerate tautomers (a separate, RDKit-backed command exists —
  `state enumerate-tautomers` — but is outside this tutorial; see the main
  [README](../README.md#what-is-implemented-today)).
- It did not dock anything, define a search space, or rank any pose.
- It did not prepare, inspect, or touch any receptor — that is a separate
  pipeline (`molecular receptor --help`), unrelated to ligand acquisition.
- It did not use any AI or automated decision-making to pick a structure,
  a state, or a conformer — every id and every choice above was named
  explicitly by the person running the command.

## Values that change on every run

Four values in this tutorial's output **may or may not match** what is
printed in this document — copy them from *your own* terminal, never
assume they are reproduced verbatim here:

| Value | Printed by | Why it varies |
|---|---|---|
| `Acquired at` | `ligand show` | Wall-clock timestamp of your own download — this one will always differ from the document. |
| RAW checksum (`Checksum:` / `RAW checksum verified:` / `Representation checksum:`) | `ligand show`, `ligand promote`, `state show` | SHA-256 is deterministic: the exact same bytes always hash to the exact same digest. The digest varies here only because PubChem's returned bytes themselves are not always identical. Confirmed directly while preparing this tutorial: the 2D SDF for CID 2519 embeds a generation timestamp in its header line (e.g. `-OEChem-10152613152D`, decoding to a date and time); two downloads requested seconds apart were byte-identical, while downloads requested several minutes apart differed. All three "checksum" lines for the *same* promotion must match each other, but may differ from the exact digits printed in this document — and may also happen to match them, if your download lands on the same generation window. |
| `Attempt ID` | `ligand promote` | A fresh random id generated for each attempt, for audit correlation only — never a scientific identifier. Always differs. |
| `Promotion ID` | `ligand promote` | **Not** a hash of the RAW content alone. It is a deterministic, versioned semantic key (`semantic_key_algorithm_version`) computed over: the ligand's own `input_id`, the RAW snapshot's checksum, the RDKit engine's `engine_name`/`engine_version`, and exactly eight named fields of the promotion policy — `sanitize`, `remove_hs`, `strict_parsing`, `record_policy`, `component_policy`, `charge_policy`, `stereo_policy`, and `representation_policy` (the policy's own `schema_version` and `source` fields do **not** participate). The same inputs always produce the same key. It changes whenever any one of those inputs changes — including the checksum above — but may stay the same across two separate runs if all of those inputs happen to match (for example, if you reuse this tutorial's exact ids and the checksum above happens to coincide too). |

Everything else in this tutorial's output — banners, labels, paths relative
to the experiment directory, the RDKit engine version — is exactly what
this exact version of the code prints, every time.

## Alternative examples

`ligand fetch-pubchem` → `ligand show` (steps 2–3 above) run the same
acquisition-and-registration path for any PubChem compound, with no
*chemistry* eligibility check of any kind — but "any compound" still
depends on PubChem actually returning an acceptable response for that CID,
and on the download completing registration successfully (see
[Troubleshooting](#troubleshooting) below for what a rejected CID looks
like). Going further to `ligand promote` (step 4) is **conditioned on the
SDF actually being eligible for promotion**: the file must contain exactly
one physical record, that
record must parse as a valid, non-null molecule, and it must be a single
connected fragment — `LigandPromotionPolicy.component_policy` always
rejects a disconnected structure (for example, a salt written as two
separate fragments within one SDF block). A compound that fails any of
these checks registers fine in step 2 but fails at the `ligand promote`
step with a specific, named error (`PromotionRecordCountError`,
`InvalidPromotionSDFError`, or `DisconnectedStructureError`).

The four CIDs below were chosen to also illustrate two identity pitfalls
worth knowing about before you pick a CID of your own.

| Compound | PubChem CID | Verified via PubChem's own public API |
|---|---|---|
| Rosmarinic acid | `5281792` | `C18H16O8`; isomeric SMILES carries an explicit `(2R)` stereocenter — this is the stereochemically-defined record, not a generic/racemic one. |
| Imidacloprid | `86287518` | `C9H10ClN5O2`; isomeric SMILES carries an explicit configuration on the nitroimine double bond. **Use this exact CID.** A nearby-numbered CID, `86287`, is a *completely unrelated compound* (a bipyridine carbonitrile) — PubChem CIDs are not sequential by substance family, and a typo here silently fetches the wrong molecule with no error of any kind. |
| Diclofenac | `3033` | `C14H11Cl2NO2` — this is the **free acid**. Sodium and potassium diclofenac (common pharmaceutical salt forms) are *different PubChem records* with different CIDs and different formulas; this tutorial does not use either salt form. |
| Atrazine | `2256` | `C8H14ClN5`. |

Fetching and registering each (verified against this repository's code
while preparing this tutorial):

```powershell
molecular ligand fetch-pubchem caffeine-demo rosmarinic-acid --cid 5281792 --name "Rosmarinic acid"
molecular ligand fetch-pubchem caffeine-demo imidacloprid --cid 86287518 --name "Imidacloprid"
molecular ligand fetch-pubchem caffeine-demo diclofenac --cid 3033 --name "Diclofenac"
molecular ligand fetch-pubchem caffeine-demo atrazine --cid 2256 --name "Atrazine"
```

Each prints `Ligand '<id>' registered successfully.`, exactly like step 2
above — confirmed by running all four `fetch-pubchem` calls against this
repository's code while preparing this tutorial. **`ligand promote` was
not separately run for these four compounds** as part of this validation —
only their registration (steps 2–3's pattern) was exercised. `ligand show`
works identically to step 3 for any of them. `ligand promote`,
`entity show`, and `state show` follow the same contract described in
steps 4–6, and the eligibility requirements described above apply equally,
but actually promoting any specific one of these four was not itself
executed here — if you run it yourself and get a named eligibility error
instead of success, that is the platform doing its job, not a bug in this
tutorial.

If you want a CID for a different compound, confirm it independently
against PubChem's own public record
(`https://pubchem.ncbi.nlm.nih.gov/compound/<cid>`) before using it here —
this tutorial's own verification above used PubChem's public
`property/Title,IUPACName,MolecularFormula,IsomericSMILES,InChIKey` API
endpoint for exactly this reason.

## Troubleshooting

An invalid or nonexistent CID fails before any registration is attempted —
confirmed directly:

```powershell
molecular ligand fetch-pubchem caffeine-demo bad-cid-test --cid 999999999999 --name "Nonexistent"
```

```
ACQUISITION ERROR
PubChem request failed with HTTP 400 for CID 999999999999.
```

(A CID that is syntactically plausible but simply does not exist reports
`PubChem has no record for CID <cid>.` instead — both cases leave nothing
registered.)
