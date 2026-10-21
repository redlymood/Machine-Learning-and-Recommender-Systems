"""Verify the HW6 references against their registries and record the URL status codes.

DOIs: Crossref REST API (api.crossref.org/works/<DOI>). [2] is Brin and Page (1998),
which replaced the Stanford PageRank report because the Stanford InfoLab server
returned only an error page (2026-10-21). TensorFlow.js:
the arXiv abstract page (citation_* meta tags; the arXiv API rate-limits), DataCite (arXiv DOI) and the official MLSys proceedings page
(citation_* meta tags; dblp is behind a bot check). Status
codes: the first response without following redirects, and the final one after
following them (publisher sites may answer 403 to non-browser clients). No
citation text from the course readme is used.

Usage: python3 refs_check.py   (writes results/references_raw.json, results/references.txt)
"""
import html
import json
import time
import re
import urllib.error
import urllib.parse
import urllib.request

from paths import RESULTS

UA = {"User-Agent": "hw6-reference-check/1.0 (python urllib)"}
DOIS = {
    "[1]": "10.1145/2827872",
    "[2]": "10.1016/S0169-7552(98)00110-X",
    "[3]": "10.1145/511446.511513",
    "[4]": "10.1145/775152.775191",
    "[5]": "10.1109/TKDE.2007.46",
    "[6]": "10.1109/MC.2009.263",
}
ARXIV_ID = "1901.05350"
TFJS_API = "https://js.tensorflow.org/api/4.22.0/"
MLSYS = ("https://proceedings.mlsys.org/paper_files/paper/2019/hash/"
         "acd593d2db87a799a8d3da5a860c028e-Abstract.html")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def status(url):
    first = final = None
    try:
        first = urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, headers=UA), timeout=30).status
    except urllib.error.HTTPError as e:
        first = e.code
    except Exception as e:  # noqa: BLE001 - report network errors as text
        first = type(e).__name__
    try:
        final = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).status
    except urllib.error.HTTPError as e:
        final = e.code
    except Exception as e:  # noqa: BLE001
        final = type(e).__name__
    return first, final


def get(url, tries=3):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(5)  # polite back-off (arXiv asks for a few seconds between requests)


def initials(given):
    parts = re.split(r"([\s-]+)", given.strip())
    return "".join(p[0] + "." if p.strip(" -") else ("-" if "-" in p else " ") for p in parts).replace(". ", ". ")


def ieee_authors(names):
    names = [f"{initials(g)} {f}" for g, f in names]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def crossref(ref, doi):
    m = json.loads(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    date = (m.get("published-print") or m.get("published") or m.get("issued"))["date-parts"][0]
    rec = {"type": m["type"], "authors": [(a.get("given", ""), a["family"]) for a in m.get("author", [])],
           "title": m["title"][0], "subtitle": (m.get("subtitle") or [None])[0],
           "published_online": (m.get("published-online") or {}).get("date-parts", [None])[0],
           "published_print": (m.get("published-print") or {}).get("date-parts", [None])[0], "container": (m.get("container-title") or [""])[0],
           "short_container": (m.get("short-container-title") or [""])[0], "volume": m.get("volume"),
           "issue": m.get("issue"), "pages": m.get("page"), "date": date, "doi": m["DOI"],
           "publisher": m.get("publisher"), "event": (m.get("event") or {}).get("name"),
           "event_location": (m.get("event") or {}).get("location"), "article_number": m.get("article-number")}
    return rec


def metas(page, prefix):
    out = {}
    for tag in re.findall(r"<meta\b[^>]*>", page):
        name, content = re.search(r'name="([^"]+)"', tag), re.search(r'content="([^"]*)"', tag)
        if name and content and name.group(1).startswith(prefix):
            out.setdefault(name.group(1), []).append(html.unescape(content.group(1)))
    return out


def tfjs():
    arxiv = metas(get(f"https://arxiv.org/abs/{ARXIV_ID}"), "citation_")
    datacite = json.loads(get(f"https://api.datacite.org/dois/10.48550/arxiv.{ARXIV_ID}"))["data"]["attributes"]
    mlsys = {}
    for name, content in re.findall(r'<meta\s+name="(citation_[^"]+)"\s+content="([^"]*)"', get(MLSYS)):
        mlsys.setdefault(name, []).append(html.unescape(content))
    return {"arxiv": arxiv,
            "datacite": {"doi": datacite["doi"], "titles": datacite["titles"], "year": datacite["publicationYear"],
                         "creators": [c["name"] for c in datacite["creators"]]},
            "mlsys": mlsys}


def main():
    raw, lines = {}, ["Reference verification (registry data; no citation text from the course readme)"]
    for ref, doi in DOIS.items():
        rec = crossref(ref, doi)
        rec["status_doi_org"] = status(f"https://doi.org/{doi}")
        raw[ref] = rec
        lines.append(f"{ref} Crossref {doi}: {json.dumps(rec, ensure_ascii=False)}")
    raw["[7]"] = tfjs()
    for key, url in (("arxiv_abs", f"https://arxiv.org/abs/{ARXIV_ID}"),
                     ("doi_org", f"https://doi.org/10.48550/arXiv.{ARXIV_ID}")):
        raw["[7]"][f"status_{key}"] = status(url)
    raw["[7]"]["status_mlsys"] = status(MLSYS)
    lines.append(f"[7] {json.dumps(raw['[7]'], ensure_ascii=False)}")
    raw["[8]"] = {"url": TFJS_API, "status": status(TFJS_API)}
    lines.append(f"[8] TF.js API: {json.dumps(raw['[8]'])}")
    (RESULTS / "references_raw.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    text = "\n".join(lines) + "\n"
    (RESULTS / "references.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
