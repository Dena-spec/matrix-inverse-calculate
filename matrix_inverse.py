# Name : Dena 데나
# Project : Matrix inverse

from fractions import Fraction


# Find the determinant. For larger matrices, expand along the first row.
def determinant(a):
    """Calculate determinant recursively using cofactor expansion."""
    n = len(a)
    if n == 1:
        return a[0][0]
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return sum((-1) ** j * a[0][j] * determinant(
        [row[:j] + row[j + 1:] for row in a[1:]])
        for j in range(n))


# Method 1: inverse = adjugate / determinant.
# Fractions help avoid rounding errors during division.
def inverse_by_determinant(a):
    a = [[Fraction(x) for x in row] for row in a]
    n = len(a)
    det = determinant(a)
    if det == 0:
        raise ValueError('Inverse does not exist (determinant = 0).')
    if n == 1:
        return [[1 / det]]
    # Build the cofactor matrix, then transpose it when returning the inverse.
    cofactors = [
        [(-1) ** (i + j) * determinant(
            [row[:j] + row[j + 1:] for k, row in enumerate(a) if k != i])
         for j in range(n)] for i in range(n)
    ]
    return [[cofactors[j][i] / det for j in range(n)] for i in range(n)]


# Method 2: change [A | I] into [I | inverse] using row operations.
def inverse_by_gauss_jordan(a, show_steps=False):
    n = len(a)
    augmented = [[Fraction(x) for x in row] +
                 [Fraction(int(i == j)) for j in range(n)]
                 for i, row in enumerate(a)]
    if show_steps:
        print_matrix('Step 0: [A | I]', augmented)
    # Count and display each row operation only when requested.
    step = 0
    def log(action):
        nonlocal step
        if show_steps:
            step += 1
            print_matrix(f'Step {step}: {action}', augmented)
    # Work through one pivot column at a time.
    for col in range(n):
        pivot = next((r for r in range(col, n) if augmented[r][col] != 0), None)
        if pivot is None:
            raise ValueError('Inverse does not exist (no nonzero pivot).')
        # Swap rows if the current pivot is zero.
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
            log(f'Swap R{col+1} and R{pivot+1}')
        # Make the pivot equal to 1.
        factor = augmented[col][col]
        augmented[col] = [x / factor for x in augmented[col]]
        log(f'Divide R{col+1} by {factor}')
        for r in range(n):
            if r != col:
                factor = augmented[r][col]
                augmented[r] = [x - factor * y for x, y in zip(augmented[r], augmented[col])]
                if factor != 0:
                    log(f'R{r+1} <- R{r+1} - ({factor})*R{col+1}')
    return [row[n:] for row in augmented]


# Multiply two matrices to check the inverse.
def multiply(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n))
             for j in range(n)] for i in range(n)]


# Check whether the product is the identity matrix.
def is_identity(a):
    return all(a[i][j] == int(i == j)
               for i in range(len(a)) for j in range(len(a)))


# Print each row so the results are easier to read.
def print_matrix(title, a):
    print(title)
    for row in a:
        print('  [ ' + '  '.join(f'{str(x):>7}' for x in row) + ' ]')


# Run both methods and compare their answers.
def run_calculation(a, show_steps=False):
    print_matrix('\nInput matrix:', a)
    det = determinant(a)
    print(f'\nDeterminant: {det}')
    try:
        result_det = inverse_by_determinant(a)
        result_gj = inverse_by_gauss_jordan(a, show_steps=show_steps)
    except ValueError as error:
        print(f'ERROR: {error}')
        print('Both methods cannot produce an inverse for this matrix.')
        return
    print_matrix('\nInverse using determinant (adjugate):', result_det)
    print_matrix('\nInverse using Gauss-Jordan elimination:', result_gj)
    print(f'\nComparison: {"IDENTICAL" if result_det == result_gj else "DIFFERENT"}')
    print_matrix('\nExtra verification - A x A_inverse:', multiply(a, result_det))
    print(f'Identity verification: {"PASS" if is_identity(multiply(a, result_det)) else "FAIL"}')


# Get the matrix from the user (integer entries only).
def main():
    print('=== MATRIX INVERSE CALCULATOR ===')
    while True:
        try:
            n = int(input('Enter matrix size n (positive integer): '))
            if n < 1:
                raise ValueError
            break
        except ValueError:
            print('Please enter a positive integer.')
    matrix = []
    for i in range(n):
        while True:
            try:
                values = input(f'Enter row {i + 1} ({n} numbers separated by spaces): ').split()
                if len(values) != n:
                    raise ValueError('Incorrect number of entries.')
                matrix.append([Fraction(int(value)) for value in values])
                break
            except (ValueError, ZeroDivisionError) as error:
                print(f'Invalid row (enter integers only): {error}. Try again.')
    # Optional: show the Gauss-Jordan process for the report.
    show_steps = input('Show Gauss-Jordan steps? (y/N): ').strip().lower() == 'y'
    run_calculation(matrix, show_steps=show_steps)


if __name__ == '__main__':
    main()
