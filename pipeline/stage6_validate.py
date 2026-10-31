"""Stage 6: Geçerlilik kontrolü.

Okunan verilen rakamlar sudoku kurallarına uyuyor mu? Hiçbir satırda, sütunda ve 3x3 kutuda
aynı rakam iki kez olmamalı. Çakışan hücreler debug görüntüsünde işaretlenmek üzere döndürülür.
"""


def find_conflicts(grid):
    """Kuralı bozan hücrelerin (satır, sütun) kümesini döndürür. 0 = boş, kontrol edilmez."""
    conflicts = set()
    groups = []
    for i in range(9):
        groups.append([(i, c) for c in range(9)])                     # satır
        groups.append([(r, i) for r in range(9)])                     # sütun
    for br in range(3):
        for bc in range(3):
            groups.append([(3 * br + r, 3 * bc + c) for r in range(3) for c in range(3)])  # kutu
    for cells in groups:
        seen = {}
        for r, c in cells:
            v = grid[r][c]
            if v == 0:
                continue
            if v in seen:
                conflicts.add((r, c))
                conflicts.add(seen[v])
            else:
                seen[v] = (r, c)
    return conflicts


def validate(grid):
    """('valid' ya da 'invalid', çakışan hücreler listesi)."""
    conflicts = find_conflicts(grid)
    return ("invalid" if conflicts else "valid"), sorted(conflicts)