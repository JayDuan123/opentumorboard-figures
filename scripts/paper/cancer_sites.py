#!/usr/bin/env python3
"""Rule-based PRIMARY-SITE labelling of the free-text [ diagnosis ] field.

Two stages, because one alone gets it wrong:

  1. strip_secondary() removes clauses that describe where the disease SPREAD.
     Without this the site of metastasis wins over the site of origin whenever it
     happens to sit higher in the rule order - 'small cell lung carcinoma with brain
     metastases' lands in CNS. A distribution of cancer types must count origins.
  2. classify() applies ordered regexes to what is left. First match wins, so the
     rule order IS the taxonomy: specific entities before the generic organ words
     they contain ('gastric MALT lymphoma' is haematologic, not upper GI).

Every unmatched string is printed rather than swept into 'other': an unaudited
bucket is how a made-up distribution hides.
"""
import re, sys, collections, json

# Clause describes spread/extension, not origin. Dropped before classification.
SECONDARY = re.compile(
    r"\b(metastas[ei]|metastatic (disease|involvement|spread|deposits?)\s+(to|in|at)"
    r"|extension (into|onto|to|through)|invasion (into|of the orbit)|spread to"
    r"|disseminat|drop metastas|leptomeningeal (spread|dissemination))", re.I)

# Units that never carry the primary even though they lack a SECONDARY keyword.
def split_units(text):
    parts = re.split(r"(?<=[.;])\s+", text)
    out = []
    for p in parts:
        out.extend(re.split(r",?\s+(?=with\b)|,?\s+(?=and (?:later|subsequent))"
                            r"|,?\s+(?=subsequent(?:ly)?\b)|,?\s+(?=later\b)", p))
    return [u for u in (s.strip() for s in out) if u]

def strip_secondary(text):
    kept = [u for u in split_units(text) if not SECONDARY.search(u)]
    return " ".join(kept) if kept else text

# (site, regex) -- ORDERED. First match wins.
RULES = [
    ("Hematologic", r"\b(leukemi|lymphom|myelom|myelodysplas|myeloprolifer|CLL|CML|AML|DLBCL|Hodgkin|plasmacytoma|myelofibrosis|Waldenstr|hairy cell|MGUS|Langerhans cell histiocytosis|amyloidosis)"),
    ("Skull base", r"\b(chordom|clival|clivus|petrous apex|jugular foramen|paragangliom|glomus (tumor|jugulare|tympanicum)|esthesioneuroblastom|olfactory neuroblastom)"),
    ("CNS / brain & spine", r"\b(glioblastom|gliom|astrocytom|astrocytic|oligodendroglio|ependymom|medulloblastom|meningiom|craniopharyngiom|pituitary|adenohypophys|schwannom|vestibulocochlear|acoustic neurom|germinom|pineal|hemangioblastom|choroid plexus|DIPG|neurocytom|gangliogliom|CNS WHO|neurofibromatosis type 2|solitary fibrous tumor|pilocytic|colloid cyst|epidermoid cyst|third ventric|ventricular mass|intracranial (mass|lesion)|hemispheric|sellar|suprasellar|cerebell(ar|opontine) (mass|lesion|tumor)|hydrocephalus|arteriovenous malformation|cavernoma|spinal (cord|anastomosing|extra-axial))"),
    ("Thyroid / parathyroid", r"\b(thyroid|thyroglossal|parathyroid|hyperparathyroid|Hashimoto|Graves|Bethesda|TI-RADS|H(ü|u)rthle)"),
    ("Head & neck", r"\b(oral cavity|tongue|larynx|laryngeal|hypopharyn|oropharyn|nasopharyn|sinonasal|paranasal|maxillary sinus|ethmoid|frontal sinus|nasal (cavity|septum|floor)|tonsil|salivary|parotid|submandibular|sublingual|buccal|palate|gingiv|floor of mouth|piriform|supraglottic|subglottic|glottic|vocal (cord|fold)|mandible|mandibular|TMJ|temporomandibular|lip\b|caudal septum|collumellar|head and neck|adenoid cystic|mucoepidermoid|neck (mass|node)|retromolar|alveolar ridge|thyroglossal)"),
    ("Breast", r"\b(breast|DCIS|ductal carcinoma in situ|mammar|invasive lobular carcinoma)"),
    ("Peritoneum", r"\b(peritoneal mesotheliom|pseudomyxom)"),
    ("Thoracic / lung", r"\b(lung|pulmonary (adenocarcinom|nodule|carcinoid|malignan)|non-small cell|NSCLC|small cell lung|SCLC|mesotheliom|pleural|thymom|thymic|bronchi|mediastin|tracheal|hilar (mass|lesion))"),
    ("Colorectal / anal", r"\b(colon|colorect|rectal|rectum|sigmoid|cecum|caecum|appendice|appendix|anal canal|anus\b)"),
    ("Upper GI", r"\b(esophag|oesophag|gastric|stomach|gastroesophageal|GEJ|duoden|jejun|ileum|ileal|small bowel|small intestin|midgut|GIST|gastrointestinal stromal)"),
    ("Hepatobiliary / pancreas", r"\b(pancrea|hepatocellular|HCC\b|hepatoblastom|tumor of the liver|liver (primary|cancer)|cholangiocarcinom|biliary|gallbladder|bile duct|ampullar|Klatskin)"),
    ("Gynecologic", r"\b(cervix|cervical (squamous|adenocarc|intraepith|cancer|lesion)|endometri|uterin|uterus|ovar(y|ies|ian)|adnexal|fallopian|vulva|vagin|gestational trophoblast|FIGO)"),
    ("Peritoneum", r"\b(periton|pseudomyxom|omental|intra-abdominal/pelvic)"),
    ("Adrenal / other endocrine", r"\b(pheochromocytom|adrenocortical|adrenal (mass|tumou?r|primary|carcinom|adenoma|nodule))"),
    ("Genitourinary", r"\b(renal|kidney|nephr|urotheli|bladder|prostat|testic|testis|seminom|penile|ureter|urethra|Wilms)"),
    ("Skin / melanoma", r"\b(melanom|cutaneous|skin\b|basal cell carcinom|Merkel|dermatofibrosarcom|keratinocyt)"),
    ("Sarcoma (bone / soft tissue)", r"\b(sarcom|osteosarcom|chondrosarcom|chondromesenchymal|Ewing|rhabdomyo|liposarcom|leiomyosarcom|desmoid|synovial|fibromatos|giant cell tumor of bone|neurofibrom|plexiform|nerve sheath|hemangiom)"),
    ("Neuroendocrine (site NOS)", r"\b(neuroendocrine|carcinoid|NET\b|pNET|neuroblastom|PNET|primitive neuroectodermal)"),
    ("Skull base", r"\bskull base\b"),
    ("Unknown primary", r"\b(unknown primary|occult primary|carcinoma of unknown|CUP\b|primary (site|tumor type) (was )?(not|un)(identified|specified|known)|not (identified|specified)|solid malignancy|malignancy in a)"),
]
COMPILED = [(s, re.compile(p, re.I)) for s, p in RULES]

def classify(text):
    """Prefer the primary-only text; fall back to the full string if that is empty."""
    for probe in (strip_secondary(text), text):
        for site, rx in COMPILED:
            m = rx.search(probe)
            if m:
                return site, m.group(0)
    return "UNMATCHED", ""


def site_counts(case_summaries):
    """(Counter of primary sites, list of per-case audit rows) over free-text summaries."""
    import collections, re
    counts, audit = collections.Counter(), []
    for meta, summary in case_summaries:
        m = re.search(r"\[\s*diagnosis\s*\]\s*:(.*?)(?=\n\s*\[|\Z)", summary, re.S | re.I)
        if not m:
            raise SystemExit(f"cancer_sites: no [ diagnosis ] section for {meta}")
        diag = " ".join(m.group(1).split())
        site, trigger = classify(diag)
        if site == "UNMATCHED":
            raise SystemExit(f"cancer_sites: unmatched diagnosis for {meta}: {diag[:120]}")
        counts[site] += 1
        audit.append({**meta, "site": site, "trigger": trigger, "diagnosis": diag})
    return counts, audit
