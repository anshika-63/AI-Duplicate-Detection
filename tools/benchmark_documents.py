"""
Document Similarity Benchmark Utility
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Evaluates TF-IDF cosine similarity, cross-format document matching,
and duplicate categorization across 9 controlled benchmark pairs.
Exports empirical results to reports/document_similarity_benchmark.csv.
"""

import csv
import os
import shutil
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from document_detection import (
    DEFAULT_DOC_SIMILARITY_THRESHOLD,
    calculate_tfidf_similarity,
    categorize_document_duplicate,
    compare_two_documents,
    extract_document_text,
    extract_document_text_with_metadata,
)


def create_minimal_docx(text: str, output_path: Path):
    """Creates a minimal valid DOCX file using python standard library zipfile."""
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )

    paragraphs = text.split("\n")
    p_xml = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs if p.strip())

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{p_xml}</w:body>'
        '</w:document>'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as docx_zip:
        docx_zip.writestr("[Content_Types].xml", content_types_xml)
        docx_zip.writestr("_rels/.rels", rels_xml)
        docx_zip.writestr("word/document.xml", document_xml)


def run_document_benchmark(output_csv: Path) -> Dict[str, float]:
    """Runs the document similarity benchmark across 9 realistic scenarios."""
    temp_dir = Path(tempfile.mkdtemp(prefix="doc_bench_"))

    try:
        # Base document paragraphs
        p1 = "Artificial intelligence duplicate detection provides scalable indexing for large-scale enterprise repositories."
        p2 = "Perceptual hashing techniques map visual images to compact binary vectors resistant to minor rotations."
        p3 = "Natural language processing vectorizers extract lexical frequencies using Term Frequency Inverse Document Frequency."
        p4 = "Deep learning convolutional models identify document layouts into sixteen standardized categories."
        p5 = "Relational SQLite storage indexes historical comparisons with timestamped audit trails."
        p6 = "The system architecture enables automated deduplication workflows across mixed file formats."

        base_text = "\n".join([p1, p2, p3, p4, p5, p6])

        # Scenario 1: Exact duplicate text (TXT)
        f_base_txt = temp_dir / "base.txt"
        f_base_txt.write_text(base_text, encoding="utf-8")

        f_exact_txt = temp_dir / "exact_copy.txt"
        f_exact_txt.write_text(base_text, encoding="utf-8")

        # Scenario 2: Cross-format identical text (DOCX)
        f_cross_docx = temp_dir / "cross_format.docx"
        create_minimal_docx(base_text, f_cross_docx)

        # Scenario 3: Minor modification (~95-98%, 1 token changed)
        p3_slight = "Natural language processing tokenizers extract lexical frequencies using Term Frequency Inverse Document Frequency."
        slight_text = "\n".join([p1, p2, p3_slight, p4, p5, p6])
        f_slight_txt = temp_dir / "minor_mod.txt"
        f_slight_txt.write_text(slight_text, encoding="utf-8")

        # Scenario 4: Paragraph reordering (shuffled order of identical content)
        reordered_text = "\n".join([p4, p1, p6, p5, p2, p3])
        f_reordered_txt = temp_dir / "reordered.txt"
        f_reordered_txt.write_text(reordered_text, encoding="utf-8")

        # Scenario 5: Moderate modification (1 sentence paraphrased, ~85-93%)
        p2_mod = "Perceptual hashing algorithms map visual images to compact binary vectors resistant to image transformations."
        mod_text = "\n".join([p1, p2_mod, p3, p4, p5, p6])
        f_moderate_txt = temp_dir / "moderate_mod.txt"
        f_moderate_txt.write_text(mod_text, encoding="utf-8")

        # Scenario 6: Borderline possible duplicate (2 sentences rewritten, ~65-72%, threshold is 75%)
        p3_partial = "Statistical tokenizers extract word counts from text documents."
        borderline_text = "\n".join([p1, p2_mod, p3_partial, p4, p5, p6])
        f_borderline_txt = temp_dir / "borderline_possible.txt"
        f_borderline_txt.write_text(borderline_text, encoding="utf-8")

        # Scenario 7: Heavy modification (~30-50% similarity, majority replaced)
        p_heavy_1 = "Astrophysical spectroscopy measures photon emission lines to calculate cosmic redshift and galactic velocity."
        p_heavy_2 = "Stellar nucleosynthesis fuses hydrogen into helium within the gravitational cores of main sequence stars."
        heavy_text = "\n".join([p1, p_heavy_1, p_heavy_2, p4])
        f_heavy_txt = temp_dir / "heavy_mod.txt"
        f_heavy_txt.write_text(heavy_text, encoding="utf-8")

        # Scenario 8: Unrelated domain (Quantum Physics, expected < 15%)
        unrelated_text = (
            "Quantum field theory unifies special relativity and quantum mechanics through operator-valued fields.\n"
            "Feynman diagrams represent perturbative interactions between electrons and virtual photons.\n"
            "Non-abelian gauge symmetries dictate the electroweak and strong nuclear interactions."
        )
        f_unrelated_txt = temp_dir / "unrelated_physics.txt"
        f_unrelated_txt.write_text(unrelated_text, encoding="utf-8")

        # Scenario 9: Empty document (expected 0.0%)
        f_empty_txt = temp_dir / "empty.txt"
        f_empty_txt.write_text("", encoding="utf-8")

        benchmark_cases = [
            ("Exact Copy (Same Format TXT)", f_base_txt, f_exact_txt, True, 100.0, 100.0),
            ("Cross-Format Identical (TXT vs DOCX)", f_base_txt, f_cross_docx, True, 99.5, 100.0),
            ("Minor Modification (1 Token Changed)", f_base_txt, f_slight_txt, True, 90.0, 99.5),
            ("Paragraph Reordering (Shuffled)", f_base_txt, f_reordered_txt, True, 95.0, 100.0),
            ("Moderate Modification (1 Sentence Paraphrased)", f_base_txt, f_moderate_txt, True, 78.0, 95.0),
            ("Borderline Review (2 Sentences Rewritten)", f_base_txt, f_borderline_txt, False, 60.0, 74.9),
            ("Heavy Modification (Majority Replaced)", f_base_txt, f_heavy_txt, False, 25.0, 55.0),
            ("Unrelated Domain (NLP vs Quantum Physics)", f_base_txt, f_unrelated_txt, False, 0.0, 15.0),
            ("Empty Document vs Base", f_base_txt, f_empty_txt, False, 0.0, 0.0),
        ]

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        records = []
        tp, fp, tn, fn = 0, 0, 0, 0

        for case_name, doc1, doc2, ground_truth_dup, exp_min, exp_max in benchmark_cases:
            res = compare_two_documents(doc1, doc2, threshold=DEFAULT_DOC_SIMILARITY_THRESHOLD)
            meta1 = extract_document_text_with_metadata(doc1)
            meta2 = extract_document_text_with_metadata(doc2)

            predicted_dup = res.is_duplicate

            if ground_truth_dup and predicted_dup:
                tp += 1
                status = "PASS"
            elif not ground_truth_dup and not predicted_dup:
                tn += 1
                status = "PASS"
            elif not ground_truth_dup and predicted_dup:
                fp += 1
                status = "FAIL (False Positive)"
            else:
                fn += 1
                status = "FAIL (False Negative)"

            in_range = exp_min <= res.similarity_score <= exp_max
            if not in_range:
                status += f" (Score {res.similarity_score:.2f}% out of [{exp_min}, {exp_max}])"

            records.append({
                "benchmark_case": case_name,
                "doc1_name": doc1.name,
                "doc2_name": doc2.name,
                "similarity_score": res.similarity_score,
                "predicted_category": res.duplicate_type,
                "predicted_duplicate": predicted_dup,
                "ground_truth_duplicate": ground_truth_dup,
                "status": status,
                "doc1_word_count": meta1.word_count,
                "doc2_word_count": meta2.word_count,
                "doc1_quality": meta1.quality_score,
                "doc2_quality": meta2.quality_score,
            })

        total = len(benchmark_cases)
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print("\n" + "=" * 65)
        print("T243 DOCUMENT SIMILARITY BENCHMARK RESULTS")
        print("=" * 65)
        print(f"Total Cases Evaluated: {total}")
        print(f"Accuracy:  {accuracy * 100:.2f}% ({tp + tn}/{total})")
        print(f"Precision: {precision:.4f} (TP: {tp}, FP: {fp})")
        print(f"Recall:    {recall:.4f} (TP: {tp}, FN: {fn})")
        print(f"F1-Score:  {f1:.4f}")
        print("-" * 65)
        for r in records:
            print(f"[{r['status'][:4]}] {r['benchmark_case']:<44}: {r['similarity_score']:>6.2f}% -> {r['predicted_category']}")
        print("=" * 65)
        print(f"Report saved to: {output_csv}\n")

        return {
            "total": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    report_path = Path("E:/AI-Duplicate-Detection/reports/document_similarity_benchmark.csv")
    run_document_benchmark(report_path)
