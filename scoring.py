"""Versioned answer extraction; retain legacy mode for reference comparisons."""
import re


def extract_answer(text, dataset, mode='strict'):
    if dataset.lower() == 'aqua':
        if mode == 'legacy':
            matches = re.findall(r'[A-E]', text)
            return matches[0] if matches else None
        explicit = re.findall(r'(?:final\s+answer|answer)(?:\s+is|\s*[:=])?\s*\(?([A-E])\)?\b', text, re.I)
        if explicit:
            return explicit[-1].upper()
        standalone = re.fullmatch(r'\s*\(?([A-E])\)?[.\s]*', text, re.I)
        return standalone.group(1).upper() if standalone else None
    text = text.replace(',', '')
    if mode == 'legacy':
        values = re.findall(r'-?\d+\.?\d*', text)
    else:
        final = re.split(r'(?:final\s+answer|answer)(?:\s+is|\s*[:=])', text, flags=re.I)[-1]
        # Fractions need explicit handling, never silently interpret 1/2 as 2.
        if re.search(r'\d\s*/\s*\d', final):
            return None
        values = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', final)
    return float(values[-1]) if values else None


def correct(prediction, label, dataset):
    if prediction is None:
        return False
    if dataset.lower() == 'aqua':
        return prediction == str(label).strip().upper()
    return abs(float(label) - prediction) <= 0.001
