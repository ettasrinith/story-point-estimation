"""Verify the mapping from the assignment's section 3.3 project names to TAWOS project names.

The assignment PDF lists 39 source -> target pairs using short or informal project names
("MESOS", "Jira Cloud", "Evergreen", ...). TAWOS stores its own project names ("Apache Mesos",
"Atlassian Jira Cloud", ...). This script makes every step of that matching explicit and
machine-checked instead of leaving it as a sentence in the report:

  1. exact            the PDF label is a TAWOS project name verbatim
  2. normalised       the PDF label is a short/informal form of one specific TAWOS project
                      (vendor prefix dropped, abbreviation expanded); no judgement call
  3. approved_remap   the PDF label names a project that does not exist in TAWOS under any
                      name; it is mapped to one of the two TAWOS projects left over after
                      steps 1-2, with the instructor's approval

It asserts the result is a bijection: 39 PDF labels -> 39 distinct TAWOS projects, each used
exactly once as a source and once as a target, with no TAWOS project left unused.

Outputs (committed):
  results/project_name_mapping.csv        one row per PDF label
  results/project_mapping_validation.json  counts, assertions and the two remaps

    python tools/verify_project_mapping.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# The 39 assigned pairs, verbatim from section 3.3 of the assignment PDF.
# --------------------------------------------------------------------------------------
PAIRS = [
    ("XD", "MESOS"), ("Usergrid", "MXNet"), ("Titanium Mobile Platform", "Usergrid"),
    ("The Titanium SDK", "Alloy Framework"), ("Software Cloud", "Appcelerator Daemon"),
    ("Sawtooth", "Appcelerator Studio"), ("Nexus", "Command-Line Interface"),
    ("MXNet", "The Titanium SDK"), ("Mule APIkit", "Titanium Mobile Platform"),
    ("Mule", "C++ driver (Appcelerator)"), ("Moodle", "Atlassian Software Server"),
    ("MongoDB Core Server", "Bamboo"), ("MESOS", "Clover"),
    ("Lyrasis Dura Cloud", "Confluence Cloud"), ("Lsstcorp Data management", "Confluence Server"),
    ("Jira Server", "Crowd"), ("Jira Cloud", "FishEye"), ("Java driver", "Jira Cloud"),
    ("Indy SDK", "Jira Server"), ("Indy Node", "Software Cloud"),
    ("FishEye", "DotNetNuke Platform"), ("Fabric", "Blockchain Explorer"),
    ("Evergreen", "Fabric"), ("DotNetNuke Platform", "Indy Node"), ("DataCass", "Indy SDK"),
    ("Crowd", "Sawtooth"), ("Confluence Server", "Lsstcorp Data management"),
    ("Confluence Cloud", "Lyrasis Dura Cloud"), ("Compass", "MongoDB Core Server"),
    ("Command-Line Interface", "Compass"), ("Clover", "C++ driver (MongoDB)"),
    ("C++ driver (Appcelerator)", "Evergreen"), ("C++ driver (MongoDB)", "Java driver"),
    ("Blockchain Explorer", "Moodle"), ("Bamboo", "Mule"),
    ("Atlassian Software Server", "Mule APIkit"), ("Appcelerator Studio", "Nexus"),
    ("Appcelerator Daemon", "DataCass"), ("Alloy Framework", "XD"),
]

# Short / informal PDF label -> the one TAWOS project it denotes. Mechanical only: the PDF
# drops a vendor prefix or uses an abbreviation, the TAWOS name is the full form of the same
# project. Nothing here changes which project a pair refers to.
NORMALISED = {
    "XD": ("Spring XD", "PDF short name; TAWOS stores the full project name Spring XD"),
    "MESOS": ("Apache Mesos", "PDF drops the vendor prefix"),
    "Usergrid": ("Apache Usergrid", "PDF drops the vendor prefix"),
    "MXNet": ("Apache MXNet", "PDF drops the vendor prefix"),
    "Software Cloud": ("Atlassian Software Cloud", "PDF drops the vendor prefix"),
    "Sawtooth": ("Hyperledger Sawtooth", "PDF drops the vendor prefix"),
    "Nexus": ("Sonatype Nexus", "PDF drops the vendor prefix"),
    "Bamboo": ("Atlassian Bamboo", "PDF drops the vendor prefix"),
    "Clover": ("Atlassian Clover", "PDF drops the vendor prefix"),
    "Confluence Cloud": ("Atlassian Confluence Cloud", "PDF drops the vendor prefix"),
    "Confluence Server": ("Atlassian Confluence Server", "PDF drops the vendor prefix"),
    "Jira Server": ("Atlassian Jira Server", "PDF drops the vendor prefix"),
    "Jira Cloud": ("Atlassian Jira Cloud", "PDF drops the vendor prefix"),
    "FishEye": ("Atlassian FishEye", "PDF drops the vendor prefix"),
    "Crowd": ("Atlassian Crowd", "PDF drops the vendor prefix"),
    "Java driver": ("The Mongo Java driver", "PDF short form of the Mongo Java driver project"),
    "Indy SDK": ("Hyperledger Indy SDK", "PDF drops the vendor prefix"),
    "Indy Node": ("Hyperledger Indy Node", "PDF drops the vendor prefix"),
    "Fabric": ("Hyperledger Fabric", "PDF drops the vendor prefix"),
    "Blockchain Explorer": ("Hyperledger Blockchain Explorer", "PDF drops the vendor prefix"),
    "DataCass": ("Spring DataCass", "PDF drops the vendor prefix"),
    "Compass": ("MongoDB Compass", "PDF drops the vendor prefix"),
    "C++ driver (MongoDB)": ("The Mongo C++ driver",
                             "PDF short form of the Mongo C++ driver project"),
    "Atlassian Software Server": ("Atlassian Atlassian Software Server",
                                  "TAWOS itself stores a duplicated vendor prefix; the PDF uses "
                                  "the de-duplicated form of the same project"),
}

# The two labels that name projects absent from TAWOS, and the two TAWOS projects that no other
# label can claim. Approved by the instructor.
APPROVED_REMAP = {
    "Evergreen": ("The MongoDB Engineering",
                  "no TAWOS project is named Evergreen; The MongoDB Engineering is the only TAWOS "
                  "project left unmatched, so the instructor approved this mapping"),
    "C++ driver (Appcelerator)": ("Aptana Studio",
                                  "no TAWOS project is named C++ driver (Appcelerator); Aptana "
                                  "Studio is the only TAWOS project left unmatched, so the "
                                  "instructor approved this mapping"),
}


def tawos_names() -> set[str]:
    """The 39 project names exactly as TAWOS stores them (stripped of stray whitespace)."""
    m = pd.read_csv(ROOT / "results" / "manifest.csv")
    names = set(m.project.astype(str).str.strip())
    assert len(names) == 39, len(names)
    return names


def main() -> int:
    tawos = tawos_names()
    labels = sorted({x for pair in PAIRS for x in pair})
    rows, problems = [], []

    for label in labels:
        if label in tawos:
            rows.append(dict(pdf_name=label, tawos_name=label, match_type="exact",
                             note="verbatim TAWOS project name"))
        elif label in NORMALISED and NORMALISED[label][0] in tawos:
            name, note = NORMALISED[label]
            rows.append(dict(pdf_name=label, tawos_name=name, match_type="normalised", note=note))
        elif label in APPROVED_REMAP and APPROVED_REMAP[label][0] in tawos:
            name, note = APPROVED_REMAP[label]
            rows.append(dict(pdf_name=label, tawos_name=name, match_type="approved_remap",
                             note=note))
        else:
            problems.append(f"unresolved PDF label: {label!r}")

    mapping = pd.DataFrame(rows).sort_values(["match_type", "pdf_name"])
    mapped = mapping.tawos_name.tolist()

    # bijection checks
    if len(set(mapped)) != len(mapped):
        problems.append("two PDF labels map to the same TAWOS project")
    if set(mapped) != tawos:
        problems.append(f"TAWOS projects left unused: {sorted(tawos - set(mapped))}")
    n_pairs = len(PAIRS)
    if len({p for p, _ in PAIRS}) != n_pairs or len({t for _, t in PAIRS}) != n_pairs:
        problems.append("a project is used more than once as a source or as a target")
    lookup = dict(zip(mapping.pdf_name, mapping.tawos_name))
    resolved = [(lookup[s], lookup[t]) for s, t in PAIRS]
    if len(set(resolved)) != n_pairs:
        problems.append("two resolved pairs are identical")

    counts = mapping.match_type.value_counts().to_dict()
    # TAWOS projects that no exact or normalised label can claim — this is exactly the pool the
    # two instructor-approved remaps draw from, so it must contain exactly those two names.
    leftover = sorted(tawos - set(mapping.loc[mapping.match_type != "approved_remap",
                                              "tawos_name"]))
    validation = dict(
        pdf_labels=len(labels), pairs=n_pairs, tawos_projects=len(tawos),
        match_type_counts={k: int(v) for k, v in counts.items()},
        approved_remaps=[{"pdf_name": k, "tawos_name": v[0]} for k, v in APPROVED_REMAP.items()],
        tawos_projects_left_over_after_exact_and_normalised=leftover,
        tawos_projects_claimed_left_over=["The MongoDB Engineering", "Aptana Studio"],
        bijection_holds=not problems,
        every_project_source_once=len({p for p, _ in PAIRS}) == n_pairs,
        every_project_target_once=len({t for _, t in PAIRS}) == n_pairs,
        problems=problems,
    )

    mapping.to_csv(ROOT / "results" / "project_name_mapping.csv", index=False)
    (ROOT / "results" / "project_mapping_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    print(mapping.to_string(index=False))
    print()
    print(json.dumps(validation, indent=2))
    if problems:
        print("\nFAILED:", *problems, sep="\n  - ")
        return 1
    print(f"\nOK: {len(labels)} PDF labels -> 39 distinct TAWOS projects, bijection holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
