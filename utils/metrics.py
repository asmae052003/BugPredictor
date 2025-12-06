import re
import pandas as pd

def analyze_java_code_robust(code, features_list):
    """
    Analyzes Java code using Regex to extract metrics and detect risks.
    Robust against syntax errors and missing dependencies.
    """
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith('//')]
    loc = len(lines)

    # Decision points
    if_count = len(re.findall(r'\bif\b', code))
    for_count = len(re.findall(r'\bfor\b', code))
    while_count = len(re.findall(r'\bwhile\b', code))
    switch_count = len(re.findall(r'\bswitch\b', code))
    case_count = len(re.findall(r'\bcase\b', code))
    try_count = len(re.findall(r'\btry\b', code))
    catch_count = len(re.findall(r'\bcatch\b', code))

    complexity = 1 + if_count + for_count + while_count + switch_count + case_count + try_count + catch_count

    # Methods (approximation)
    methods = len(re.findall(r'\b(public|private|protected)?\s+\w[\w\s<>\[\]]*\]]*\s+\w+\s*\(', code))

    # Method calls
    calls = len(re.findall(r'\.\w+\s*\(', code))

    # Risks
    buffer_overflow = bool(re.search(r'i\s*<=\s*\w+\.length', code, re.I)) or bool(re.search(r'i\s*<=\s*length', code, re.I))
    division_zero = "/ 0" in code or "/0" in code

    nullpointer_risk = 0
    if re.search(r'\.\w+\s*\(\s*\)\s*\.', code): 
        nullpointer_risk += 0.4
    if re.search(r'\.\w+\s*\(\s*\)\s*\.\w+\s*\(', code):
        nullpointer_risk += 0.5
    if re.search(r'\.\w+\s*\(\s*\)\s*\.\w+\s*\(\s*\)\s*\.', code):
        nullpointer_risk += 0.7
    if re.search(r'toUpperCase|toLowerCase|length|get|size|isEmpty', code, re.I):
        nullpointer_risk += 0.2
    if not re.search(r'\bif\s*\([^)]*null[^)]*\)', code):
        if "return" in code or "throw" in code:
            nullpointer_risk += 0.3

    # CK Metrics
    data = {f: 0.0 for f in features_list}
    data["loc"] = loc
    data["wmc"] = max(1, methods)
    data["v(g)"] = complexity
    data["max_cc"] = complexity
    data["avg_cc"] = complexity / max(1, methods)
    data["cbo"] = calls
    data["rfc"] = calls + methods
    data["npm"] = methods
    data["amc"] = loc / max(1, methods) if methods > 0 else loc

    return pd.DataFrame([data])[features_list], nullpointer_risk, buffer_overflow, division_zero

def analyze_cpp_code(code, features_list):
    """
    Analyzes C++ code to extract complexity and basic metrics.
    """
    lines = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith('//')])
    
    if_count = len(re.findall(r'\bif\b', code, re.I))
    for_count = len(re.findall(r'\bfor\b', code, re.I))
    while_count = len(re.findall(r'\bwhile\b', code, re.I))
    switch_count = len(re.findall(r'\bswitch\b', code, re.I))
    case_count = len(re.findall(r'\bcase\b', code, re.I))

    complexity = 1 + if_count + for_count + while_count + switch_count + case_count
    
    # Placeholder for full features extraction if needed later
    # For now ensuring basic return for display
    return lines, complexity
