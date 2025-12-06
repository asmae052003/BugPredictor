import re
import math
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.c_cols = [
            'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
            'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
            'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'
        ]
        self.java_cols = [
            'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 
            'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc'
        ]

    def extract_metrics(self, code, language):
        """
        Extracts software metrics from the given code string.
        Returns a pandas DataFrame with a single row containing the features.
        """
        if language == 'java':
            return self._extract_java_metrics(code)
        else:
            return self._extract_cpp_metrics(code)

    def _extract_cpp_metrics(self, code):
        metrics = {}
        
        # 1. Lines of Code Metrics
        lines = code.split('\n')
        metrics['loc'] = len(lines)
        metrics['lOBlank'] = sum(1 for line in lines if not line.strip())
        metrics['lOComment'] = self._count_comments(code, 'c++')
        metrics['lOCode'] = metrics['loc'] - metrics['lOBlank'] - metrics['lOComment']
        metrics['locCodeAndComment'] = 0 
        
        # 2. Halstead Metrics
        operators, operands = self._get_operators_and_operands(code, 'c++')
        
        n1 = len(operators)
        n2 = len(operands)
        N = n1 + n2
        
        mu1 = len(set(operators))
        mu2 = len(set(operands))
        mu = mu1 + mu2
        
        if mu2 == 0: mu2 = 1
        
        metrics['n'] = N
        metrics['uniq_Op'] = mu1
        metrics['uniq_Opnd'] = mu2
        metrics['total_Op'] = n1
        metrics['total_Opnd'] = n2
        
        if mu > 0:
            metrics['v'] = N * math.log2(mu)
        else:
            metrics['v'] = 0
            
        metrics['d'] = (mu1 / 2) * (n2 / mu2)
        metrics['e'] = metrics['d'] * metrics['v']
        
        if metrics['d'] > 0:
            metrics['l'] = 1 / metrics['d']
        else:
            metrics['l'] = 0
            
        metrics['i'] = metrics['l'] * metrics['v']
        metrics['b'] = metrics['v'] / 3000
        metrics['t'] = metrics['e'] / 18
        
        # 3. McCabe
        metrics['v(g)'] = self._calculate_cyclomatic_complexity(code)
        metrics['ev(g)'] = metrics['v(g)']
        metrics['iv(g)'] = metrics['v(g)']
        metrics['branchCount'] = self._count_branches(code)

        df = pd.DataFrame([metrics])
        for col in self.c_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.c_cols]

    def _extract_java_metrics(self, code):
        # Try to use javalang if available (better accuracy)
        try:
            import javalang
            return self._extract_java_metrics_javalang(code)
        except ImportError:
            pass
        except Exception:
            pass # Fallback to regex if parsing fails

        # Fallback: Regex-based extraction (same as before)
        metrics = {}
        lines = code.split('\n')
        
        metrics['loc'] = len(lines)
        
        method_pattern = r'(public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])'
        methods = re.findall(method_pattern, code)
        metrics['wmc'] = len(methods)
        if metrics['wmc'] == 0: metrics['wmc'] = 1
        
        metrics['dit'] = 1 + len(re.findall(r'\bextends\b', code))
        metrics['noc'] = 0
        metrics['cbo'] = len(re.findall(r'\bimport\b', code))
        
        method_calls = len(re.findall(r'\.\w+\(', code))
        metrics['rfc'] = metrics['wmc'] + method_calls
        
        metrics['lcom'] = 0
        metrics['lcom3'] = 0
        metrics['ca'] = 0
        metrics['ce'] = 0
        
        metrics['npm'] = len(re.findall(r'public\s+[\w\<\>\[\]]+\s+\w+\(', code))
        
        attributes = len(re.findall(r'(private|protected)\s+[\w\<\>\[\]]+\s+\w+;', code))
        total_attrs = len(re.findall(r'(public|private|protected)\s+[\w\<\>\[\]]+\s+\w+;', code))
        metrics['dam'] = attributes / total_attrs if total_attrs > 0 else 1.0
        
        metrics['moa'] = 0
        metrics['mfa'] = 0
        metrics['cam'] = 0.5
        metrics['ic'] = 0
        metrics['cbm'] = 0
        
        metrics['amc'] = metrics['loc'] / metrics['wmc']
        
        total_cc = self._calculate_cyclomatic_complexity(code)
        metrics['avg_cc'] = total_cc / metrics['wmc']
        metrics['max_cc'] = total_cc
        
        df = pd.DataFrame([metrics])
        for col in self.java_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.java_cols]

    def _extract_java_metrics_javalang(self, code):
        import javalang
        metrics = {
            'wmc': 0, 'dit': 1, 'noc': 0, 'cbo': 0, 'rfc': 0, 'lcom': 0,
            'ca': 0, 'ce': 0, 'npm': 0, 'lcom3': 0.0, 'loc': 0,
            'dam': 1.0, 'moa': 0, 'mfa': 0.0, 'cam': 1.0,
            'ic': 0, 'cbm': 0, 'amc': 0.0, 'max_cc': 1, 'avg_cc': 1.0
        }
        
        try:
            tree = javalang.parse.parse(code)
            method_calls = set()

            for path, node in tree:
                if isinstance(node, javalang.tree.MethodDeclaration):
                    metrics['wmc'] += 1
                    if 'public' in node.modifiers:
                        metrics['npm'] += 1
                    cc = 1
                    for _, n in node.filter((javalang.tree.IfStatement, javalang.tree.ForStatement,
                                           javalang.tree.WhileStatement, javalang.tree.DoStatement,
                                           javalang.tree.SwitchStatement)):
                        cc += 1
                    metrics['max_cc'] = max(metrics['max_cc'], cc)

                if isinstance(node, javalang.tree.MethodInvocation):
                    method_calls.add(node.member)

            metrics['loc'] = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith('//')])
            metrics['rfc'] = len(method_calls) + metrics['wmc']
            metrics['cbo'] = len(method_calls)
            
            if metrics['wmc'] > 0:
                metrics['avg_cc'] = metrics['max_cc'] / metrics['wmc'] # Approximation
                metrics['amc'] = metrics['loc'] / metrics['wmc']
                
        except Exception:
            # If parsing fails, return empty/default metrics which will be filled with 0s
            pass

        df = pd.DataFrame([metrics])
        for col in self.java_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.java_cols]

    def _count_comments(self, code, language):
        single_line = len(re.findall(r'//.*', code))
        multi_line = len(re.findall(r'/\*[\s\S]*?\*/', code))
        return single_line + multi_line

    def _get_operators_and_operands(self, code, language):
        code_no_comments = re.sub(r'//.*', '', code)
        code_no_comments = re.sub(r'/\*[\s\S]*?\*/', '', code_no_comments)
        
        ops_pattern = r'[\+\-\*/%=&|<>!^~]+|\b(if|else|for|while|do|switch|case|default|return|break|continue|try|catch|throw|new)\b'
        operators = re.findall(ops_pattern, code_no_comments)
        
        opnds_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b|"[^"]*"'
        keywords = {'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'return', 'break', 'continue', 'try', 'catch', 'throw', 'new', 'int', 'float', 'double', 'char', 'void', 'class', 'public', 'private', 'protected', 'static', 'final', 'const', 'import', 'package'}
        
        raw_operands = re.findall(opnds_pattern, code_no_comments)
        operands = [op for op in raw_operands if op not in keywords]
        
        return operators, operands

    def _calculate_cyclomatic_complexity(self, code):
        decision_points = ['if', 'for', 'while', 'case', 'catch', '&&', '||', '?']
        count = 1
        for point in decision_points:
            count += code.count(point)
        return count

    def _count_branches(self, code):
        branches = ['if', 'else', 'for', 'while', 'case', 'catch']
        count = 0
        for branch in branches:
            count += code.count(branch)
        return count
