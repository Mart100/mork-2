"""Mid-difficulty Python function tasks for ``mork-eval --tasks mid``.

Harder than the smoke set (loops with conditions, small string/list/dict
algorithms, a few two-step problems), much easier than MBPP. Evaluation
only; never put these into training data.

Each task is (name, params, doc, example call, example result, asserts,
reference body). The reference bodies are only used by ``check_references``;
run ``python -m mork.eval_mid`` after editing tasks.
"""

import textwrap

MidTask = tuple[str, str, str, str, str, list[str], str]

MID_TASKS: list[MidTask] = [
    # --- strings ---
    ("count_char", "s, ch", "Return how many times the character ch appears in the string s.",
     'count_char("banana", "a")', "3",
     ["count_char('banana', 'a') == 3", "count_char('hello', 'z') == 0", "count_char('', 'a') == 0"],
     """
    return s.count(ch)
"""),
    ("reverse_words", "sentence", "Return the sentence with the order of its words reversed.",
     'reverse_words("hello big world")', "'world big hello'",
     ["reverse_words('hello big world') == 'world big hello'", "reverse_words('a b c') == 'c b a'",
      "reverse_words('single') == 'single'"],
     """
    return " ".join(sentence.split()[::-1])
"""),
    ("capitalize_words", "sentence", "Return the sentence with the first letter of every word in uppercase.",
     'capitalize_words("hello world")', "'Hello World'",
     ["capitalize_words('hello world') == 'Hello World'", "capitalize_words('a b') == 'A B'",
      "capitalize_words('python is fun') == 'Python Is Fun'"],
     """
    return " ".join(w[:1].upper() + w[1:] for w in sentence.split(" "))
"""),
    ("remove_vowels", "s", "Return s with all vowels (a, e, i, o, u, lowercase or uppercase) removed.",
     'remove_vowels("Hello World")', "'Hll Wrld'",
     ["remove_vowels('Hello World') == 'Hll Wrld'", "remove_vowels('aeiou') == ''",
      "remove_vowels('xyz') == 'xyz'", "remove_vowels('Education') == 'dctn'"],
     """
    return "".join(c for c in s if c.lower() not in "aeiou")
"""),
    ("is_anagram", "a, b", "Return True if the strings a and b contain exactly the same letters, in any order.",
     'is_anagram("listen", "silent")', "True",
     ["is_anagram('listen', 'silent') == True", "is_anagram('abc', 'cab') == True",
      "is_anagram('abc', 'abd') == False", "is_anagram('aab', 'ab') == False"],
     """
    return sorted(a) == sorted(b)
"""),
    ("longest_word", "sentence",
     "Return the longest word in the sentence. If several words are equally long, return the first one.",
     'longest_word("the quick brown fox")', "'quick'",
     ["longest_word('the quick brown fox') == 'quick'", "longest_word('a bb cc') == 'bb'",
      "longest_word('hello') == 'hello'", "longest_word('hi there you') == 'there'"],
     """
    best = ""
    for word in sentence.split():
        if len(word) > len(best):
            best = word
    return best
"""),
    ("is_pangram", "s",
     "Return True if s contains every letter of the English alphabet at least once, ignoring case.",
     'is_pangram("The quick brown fox jumps over the lazy dog")', "True",
     ["is_pangram('The quick brown fox jumps over the lazy dog') == True", "is_pangram('hello') == False",
      "is_pangram('abcdefghijklmnopqrstuvwxyz') == True", "is_pangram('ABCDEFGHIJKLMNOPQRSTUVWXY') == False"],
     """
    return set("abcdefghijklmnopqrstuvwxyz") <= set(s.lower())
"""),
    ("camel_to_snake", "name", "Convert a camelCase name to snake_case.",
     'camel_to_snake("helloWorld")', "'hello_world'",
     ["camel_to_snake('helloWorld') == 'hello_world'", "camel_to_snake('already') == 'already'",
      "camel_to_snake('myVariableName') == 'my_variable_name'", "camel_to_snake('aB') == 'a_b'"],
     """
    out = ""
    for c in name:
        if c.isupper():
            out += "_" + c.lower()
        else:
            out += c
    return out
"""),
    ("snake_to_camel", "name", "Convert a snake_case name to camelCase.",
     'snake_to_camel("hello_world")', "'helloWorld'",
     ["snake_to_camel('hello_world') == 'helloWorld'", "snake_to_camel('my_variable_name') == 'myVariableName'",
      "snake_to_camel('single') == 'single'"],
     """
    first, *rest = name.split("_")
    return first + "".join(w.capitalize() for w in rest)
"""),
    ("compress", "s",
     "Run-length encode s: each run of the same character becomes the character followed by the run length.",
     'compress("aaabcc")', "'a3b1c2'",
     ["compress('aaabcc') == 'a3b1c2'", "compress('') == ''", "compress('a') == 'a1'", "compress('zzzz') == 'z4'"],
     """
    out = ""
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out += s[i] + str(j - i)
        i = j
    return out
"""),
    ("caesar_shift", "text, k",
     "Shift every lowercase letter in text forward by k places in the alphabet, wrapping from z to a. "
     "Other characters stay the same.",
     'caesar_shift("abc xyz", 3)', "'def abc'",
     ["caesar_shift('abc xyz', 3) == 'def abc'", "caesar_shift('hello', 0) == 'hello'",
      "caesar_shift('z!', 1) == 'a!'", "caesar_shift('abc', 26) == 'abc'"],
     """
    out = ""
    for c in text:
        if "a" <= c <= "z":
            out += chr((ord(c) - ord("a") + k) % 26 + ord("a"))
        else:
            out += c
    return out
"""),
    ("alternate_case", "s",
     "Return s with characters at even indexes uppercased and characters at odd indexes lowercased.",
     'alternate_case("python")', "'PyThOn'",
     ["alternate_case('python') == 'PyThOn'", "alternate_case('') == ''", "alternate_case('AB') == 'Ab'",
      "alternate_case('abc d') == 'AbC D'"],
     """
    return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(s))
"""),
    ("count_uppercase", "s", "Return the number of uppercase letters in s.",
     'count_uppercase("Hello World")', "2",
     ["count_uppercase('Hello World') == 2", "count_uppercase('') == 0", "count_uppercase('ABC') == 3",
      "count_uppercase('abc') == 0"],
     """
    return sum(1 for c in s if c.isupper())
"""),
    ("count_substring", "s, sub", "Return how many times sub occurs in s, counting overlapping occurrences.",
     'count_substring("aaaa", "aa")', "3",
     ["count_substring('aaaa', 'aa') == 3", "count_substring('abc', 'd') == 0",
      "count_substring('abab', 'ab') == 2", "count_substring('aaa', 'a') == 3"],
     """
    return sum(1 for i in range(len(s) - len(sub) + 1) if s[i:i + len(sub)] == sub)
"""),
    ("normalize_spaces", "s",
     "Return s with each run of spaces collapsed into a single space and leading and trailing spaces removed.",
     'normalize_spaces("  hello   world ")', "'hello world'",
     ["normalize_spaces('  hello   world ') == 'hello world'", "normalize_spaces('') == ''",
      "normalize_spaces('a') == 'a'", "normalize_spaces('a  b  c') == 'a b c'"],
     """
    return " ".join(s.split())
"""),
    ("initials", "name", "Return the uppercase initials of a full name, each followed by a dot.",
     'initials("john ronald tolkien")', "'J.R.T.'",
     ["initials('john ronald tolkien') == 'J.R.T.'", "initials('ada lovelace') == 'A.L.'",
      "initials('Plato') == 'P.'"],
     """
    return "".join(w[0].upper() + "." for w in name.split())
"""),
    ("is_valid_password", "password",
     "Return True if the password has at least 8 characters, at least one digit and at least one "
     "uppercase letter.",
     'is_valid_password("Secret123")', "True",
     ["is_valid_password('Secret123') == True", "is_valid_password('short1A') == False",
      "is_valid_password('longpassword1') == False", "is_valid_password('LONGPASS') == False",
      "is_valid_password('Abcdefg1') == True"],
     """
    return (
        len(password) >= 8
        and any(c.isdigit() for c in password)
        and any(c.isupper() for c in password)
    )
"""),
    ("first_repeated_char", "s",
     "Return the first character that appears a second time while reading s from left to right, "
     "or None if no character repeats.",
     'first_repeated_char("abcbad")', "'b'",
     ["first_repeated_char('abcbad') == 'b'", "first_repeated_char('abc') is None",
      "first_repeated_char('aa') == 'a'", "first_repeated_char('abca') == 'a'"],
     """
    seen = set()
    for c in s:
        if c in seen:
            return c
        seen.add(c)
    return None
"""),
    ("first_unique_char", "s", "Return the first character in s that appears exactly once, or None if there is none.",
     'first_unique_char("aabbcdd")', "'c'",
     ["first_unique_char('aabbcdd') == 'c'", "first_unique_char('aabb') is None",
      "first_unique_char('x') == 'x'", "first_unique_char('abacb') == 'c'"],
     """
    for c in s:
        if s.count(c) == 1:
            return c
    return None
"""),
    ("most_common_char", "s",
     "Return the character that appears most often in the non-empty string s. On a tie, return the one "
     "that appears first in s.",
     'most_common_char("abracadabra")', "'a'",
     ["most_common_char('abracadabra') == 'a'", "most_common_char('abcabc') == 'a'",
      "most_common_char('zzy') == 'z'", "most_common_char('xyy') == 'y'"],
     """
    best = s[0]
    for c in s:
        if s.count(c) > s.count(best):
            best = c
    return best
"""),
    ("is_balanced", "s", "Return True if the parentheses '(' and ')' in s are balanced.",
     'is_balanced("(a(b)c)")', "True",
     ["is_balanced('(a(b)c)') == True", "is_balanced(')(') == False", "is_balanced('') == True",
      "is_balanced('(()') == False", "is_balanced('()()') == True"],
     """
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
"""),
    ("truncate", "s, n",
     "Return s unchanged if it has at most n characters, otherwise its first n characters followed by '...'.",
     'truncate("hello world", 5)', "'hello...'",
     ["truncate('hello world', 5) == 'hello...'", "truncate('hi', 5) == 'hi'", "truncate('abc', 3) == 'abc'"],
     """
    if len(s) <= n:
        return s
    return s[:n] + "..."
"""),
    ("zero_pad", "n, width",
     "Return the non-negative integer n as a string padded with leading zeros to the given width.",
     'zero_pad(42, 5)', "'00042'",
     ["zero_pad(42, 5) == '00042'", "zero_pad(7, 1) == '7'", "zero_pad(1234, 2) == '1234'",
      "zero_pad(0, 3) == '000'"],
     """
    s = str(n)
    return "0" * (width - len(s)) + s
"""),
    ("hamming_distance", "a, b", "Return the number of positions at which the equal-length strings a and b differ.",
     'hamming_distance("karolin", "kathrin")', "3",
     ["hamming_distance('karolin', 'kathrin') == 3", "hamming_distance('', '') == 0",
      "hamming_distance('abc', 'abc') == 0", "hamming_distance('abc', 'xyz') == 3"],
     """
    return sum(1 for x, y in zip(a, b) if x != y)
"""),
    ("count_char_types", "s",
     "Return a tuple (letters, digits, others) counting the letters, digits and other characters in s.",
     'count_char_types("ab12!")', "(2, 2, 1)",
     ["count_char_types('ab12!') == (2, 2, 1)", "count_char_types('') == (0, 0, 0)",
      "count_char_types('   ') == (0, 0, 3)"],
     """
    letters = sum(1 for c in s if c.isalpha())
    digits = sum(1 for c in s if c.isdigit())
    return (letters, digits, len(s) - letters - digits)
"""),
    ("word_lengths", "sentence", "Return a list with the length of each word in the sentence.",
     'word_lengths("the quick fox")', "[3, 5, 3]",
     ["word_lengths('the quick fox') == [3, 5, 3]", "word_lengths('') == []", "word_lengths('a') == [1]"],
     """
    return [len(w) for w in sentence.split()]
"""),
    ("count_words_longer_than", "sentence, n", "Return how many words in the sentence have more than n characters.",
     'count_words_longer_than("the quick brown fox", 3)', "2",
     ["count_words_longer_than('the quick brown fox', 3) == 2", "count_words_longer_than('', 1) == 0",
      "count_words_longer_than('a bb ccc', 1) == 2", "count_words_longer_than('hello world', 10) == 0"],
     """
    return sum(1 for w in sentence.split() if len(w) > n)
"""),
    ("count_vowel_words", "sentence",
     "Return how many words in the sentence start with a vowel (a, e, i, o, u), ignoring case.",
     'count_vowel_words("An apple is on the table")', "4",
     ["count_vowel_words('An apple is on the table') == 4", "count_vowel_words('') == 0",
      "count_vowel_words('Umbrella under trees') == 2"],
     """
    return sum(1 for w in sentence.split() if w[0].lower() in "aeiou")
"""),
    ("filter_long_words", "words, n", "Return the words that have at least n characters, keeping their order.",
     'filter_long_words(["a", "abc", "abcd"], 3)', "['abc', 'abcd']",
     ["filter_long_words(['a', 'abc', 'abcd'], 3) == ['abc', 'abcd']", "filter_long_words([], 2) == []",
      "filter_long_words(['hi'], 3) == []"],
     """
    return [w for w in words if len(w) >= n]
"""),
    ("acronym", "phrase", "Return the acronym of the phrase: the first letter of each word, in uppercase.",
     'acronym("portable network graphics")', "'PNG'",
     ['acronym("portable network graphics") == "PNG"',
      'acronym("as soon as possible") == "ASAP"',
      'acronym("Python") == "P"'],
     """
    return "".join(w[0].upper() for w in phrase.split())
"""),
    ("is_palindrome_sentence", "s", "Return True if s reads the same forwards and backwards, ignoring case and any character that is "
     "not a letter.",
     'is_palindrome_sentence("A man, a plan, a canal: Panama")', "True",
     ['is_palindrome_sentence("A man, a plan, a canal: Panama") == True',
      'is_palindrome_sentence("hello") == False',
      'is_palindrome_sentence("") == True'],
     """
    letters = [c.lower() for c in s if c.isalpha()]
    return letters == letters[::-1]
"""),
    ("rotate_string", "s, n", "Rotate the string s to the left by n characters.",
     'rotate_string("abcdef", 2)', "'cdefab'",
     ['rotate_string("abcdef", 2) == "cdefab"',
      'rotate_string("abc", 0) == "abc"',
      'rotate_string("abc", 3) == "abc"',
      'rotate_string("abc", 4) == "bca"'],
     """
    if not s:
        return s
    n = n % len(s)
    return s[n:] + s[:n]
"""),
    ("is_rotation", "a, b", "Return True if the string b is a rotation of the string a.",
     'is_rotation("waterbottle", "erbottlewat")', "True",
     ['is_rotation("waterbottle", "erbottlewat") == True',
      'is_rotation("abc", "acb") == False',
      'is_rotation("", "") == True',
      'is_rotation("a", "aa") == False'],
     """
    return len(a) == len(b) and b in a + a
"""),
    ("pad_left", "s, width, ch", "Return s padded on the left with the character ch until it is at least width characters long.",
     'pad_left("7", 3, "0")', "'007'",
     ['pad_left("7", 3, "0") == "007"',
      'pad_left("abc", 2, "*") == "abc"',
      'pad_left("", 2, "x") == "xx"'],
     """
    while len(s) < width:
        s = ch + s
    return s
"""),
    ("slugify", "title", "Return the title in lowercase with spaces replaced by hyphens.",
     'slugify("Hello World")', "'hello-world'",
     ['slugify("Hello World") == "hello-world"',
      'slugify("Python Is Fun") == "python-is-fun"',
      'slugify("x") == "x"'],
     """
    return title.lower().replace(" ", "-")
"""),
    ("count_lines", "text", "Return the number of non-empty lines in text.",
     'count_lines("a\\nb\\n\\nc")', "3",
     ['count_lines("a\\nb\\n\\nc") == 3',
      'count_lines("") == 0',
      'count_lines("one") == 1'],
     r"""
    return sum(1 for line in text.split("\n") if line)
"""),
    ("strip_punctuation", "s", "Return s with the characters . , ! ? ; : removed.",
     'strip_punctuation("Hello, world!")', "'Hello world'",
     ['strip_punctuation("Hello, world!") == "Hello world"',
      'strip_punctuation("no punct") == "no punct"',
      'strip_punctuation("a.b;c:d?") == "abcd"'],
     """
    return "".join(c for c in s if c not in ".,!?;:")
"""),
    ("middle_chars", "s", "Return the middle character of s if its length is odd, or the middle two characters if it is "
     "even.",
     'middle_chars("abcd")', "'bc'",
     ['middle_chars("abc") == "b"',
      'middle_chars("abcd") == "bc"',
      'middle_chars("a") == "a"'],
     """
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return s[mid - 1:mid + 1]
"""),
    ("is_digit_string", "s", "Return True if s is not empty and every character in s is a digit.",
     'is_digit_string("123")', "True",
     ['is_digit_string("123") == True',
      'is_digit_string("12a") == False',
      'is_digit_string("") == False'],
     """
    return len(s) > 0 and all(c in "0123456789" for c in s)
"""),
    ("sum_digits_in_string", "s", "Return the sum of all digit characters in s.",
     'sum_digits_in_string("a1b2c3")', "6",
     ['sum_digits_in_string("a1b2c3") == 6',
      'sum_digits_in_string("abc") == 0',
      'sum_digits_in_string("99") == 18'],
     """
    return sum(int(c) for c in s if c.isdigit())
"""),
    ("repeat_chars", "s, n", "Return a string where every character of s is repeated n times.",
     'repeat_chars("abc", 2)', "'aabbcc'",
     ['repeat_chars("abc", 2) == "aabbcc"',
      'repeat_chars("", 3) == ""',
      'repeat_chars("x", 1) == "x"'],
     """
    return "".join(c * n for c in s)
"""),
    # --- numbers ---
    ("count_digits", "n", "Return the number of digits in the non-negative integer n.",
     "count_digits(12345)", "5",
     ["count_digits(12345) == 5", "count_digits(0) == 1", "count_digits(7) == 1", "count_digits(1000) == 4"],
     """
    return len(str(n))
"""),
    ("digit_sum", "n", "Return the sum of the digits of the non-negative integer n.",
     "digit_sum(123)", "6",
     ["digit_sum(123) == 6", "digit_sum(0) == 0", "digit_sum(9999) == 36", "digit_sum(105) == 6"],
     """
    return sum(int(d) for d in str(n))
"""),
    ("digital_root", "n",
     "Repeatedly sum the digits of the non-negative integer n until a single digit remains, and return it.",
     "digital_root(9875)", "2",
     ["digital_root(9875) == 2", "digital_root(0) == 0", "digital_root(5) == 5", "digital_root(38) == 2"],
     """
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n
"""),
    ("reverse_number", "n", "Return the digits of the non-negative integer n in reverse order, as an integer.",
     "reverse_number(1234)", "4321",
     ["reverse_number(1234) == 4321", "reverse_number(0) == 0", "reverse_number(100) == 1",
      "reverse_number(7) == 7"],
     """
    return int(str(n)[::-1])
"""),
    ("is_palindrome_number", "n", "Return True if the non-negative integer n reads the same forwards and backwards.",
     "is_palindrome_number(121)", "True",
     ["is_palindrome_number(121) == True", "is_palindrome_number(10) == False",
      "is_palindrome_number(7) == True", "is_palindrome_number(1221) == True"],
     """
    return str(n) == str(n)[::-1]
"""),
    ("count_even_digits", "n", "Return how many digits of the non-negative integer n are even.",
     "count_even_digits(123456)", "3",
     ["count_even_digits(123456) == 3", "count_even_digits(0) == 1", "count_even_digits(13579) == 0",
      "count_even_digits(2468) == 4"],
     """
    return sum(1 for d in str(n) if int(d) % 2 == 0)
"""),
    ("largest_digit", "n", "Return the largest digit of the non-negative integer n.",
     "largest_digit(3829)", "9",
     ["largest_digit(3829) == 9", "largest_digit(0) == 0", "largest_digit(111) == 1", "largest_digit(507) == 7"],
     """
    return max(int(d) for d in str(n))
"""),
    ("gcd", "a, b", "Return the greatest common divisor of two positive integers.",
     "gcd(12, 18)", "6",
     ["gcd(12, 18) == 6", "gcd(7, 5) == 1", "gcd(100, 75) == 25", "gcd(9, 9) == 9"],
     """
    while b:
        a, b = b, a % b
    return a
"""),
    ("lcm", "a, b", "Return the least common multiple of two positive integers.",
     "lcm(4, 6)", "12",
     ["lcm(4, 6) == 12", "lcm(3, 5) == 15", "lcm(6, 6) == 6", "lcm(12, 18) == 36"],
     """
    x, y = a, b
    while y:
        x, y = y, x % y
    return a * b // x
"""),
    ("is_leap_year", "year", "Return True if year is a leap year in the Gregorian calendar.",
     "is_leap_year(2024)", "True",
     ["is_leap_year(2024) == True", "is_leap_year(1900) == False", "is_leap_year(2000) == True",
      "is_leap_year(2023) == False"],
     """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
"""),
    ("days_in_month", "month, year", "Return the number of days in the given month (1-12) of the given year.",
     "days_in_month(2, 2024)", "29",
     ["days_in_month(2, 2024) == 29", "days_in_month(2, 2023) == 28", "days_in_month(4, 2023) == 30",
      "days_in_month(12, 2023) == 31", "days_in_month(2, 1900) == 28"],
     """
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    if month in (4, 6, 9, 11):
        return 30
    return 31
"""),
    ("collatz_steps", "n",
     "Return how many steps the Collatz process takes to reach 1 from n: halve n if it is even, "
     "otherwise replace it with 3 * n + 1.",
     "collatz_steps(6)", "8",
     ["collatz_steps(6) == 8", "collatz_steps(1) == 0", "collatz_steps(2) == 1", "collatz_steps(7) == 16"],
     """
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps
"""),
    ("count_primes_below", "n", "Return how many prime numbers are less than n.",
     "count_primes_below(10)", "4",
     ["count_primes_below(10) == 4", "count_primes_below(2) == 0", "count_primes_below(3) == 1",
      "count_primes_below(20) == 8", "count_primes_below(100) == 25"],
     """
    count = 0
    for k in range(2, n):
        if all(k % d for d in range(2, int(k ** 0.5) + 1)):
            count += 1
    return count
"""),
    ("triangle_number", "n", "Return the sum 1 + 2 + ... + n.",
     "triangle_number(4)", "10",
     ["triangle_number(4) == 10", "triangle_number(0) == 0", "triangle_number(1) == 1",
      "triangle_number(100) == 5050"],
     """
    return n * (n + 1) // 2
"""),
    ("sum_of_squares", "n", "Return 1**2 + 2**2 + ... + n**2.",
     "sum_of_squares(3)", "14",
     ["sum_of_squares(3) == 14", "sum_of_squares(0) == 0", "sum_of_squares(1) == 1", "sum_of_squares(10) == 385"],
     """
    return sum(k * k for k in range(1, n + 1))
"""),
    ("is_perfect_square", "n", "Return True if the non-negative integer n is a perfect square.",
     "is_perfect_square(16)", "True",
     ["is_perfect_square(16) == True", "is_perfect_square(0) == True", "is_perfect_square(15) == False",
      "is_perfect_square(1) == True", "is_perfect_square(26) == False"],
     """
    r = int(n ** 0.5)
    return r * r == n or (r + 1) * (r + 1) == n
"""),
    ("is_power_of_two", "n", "Return True if n is a positive power of two (1, 2, 4, 8, ...).",
     "is_power_of_two(16)", "True",
     ["is_power_of_two(16) == True", "is_power_of_two(1) == True", "is_power_of_two(0) == False",
      "is_power_of_two(6) == False", "is_power_of_two(1024) == True"],
     """
    return n > 0 and n & (n - 1) == 0
"""),
    ("factors", "n", "Return a sorted list of all positive divisors of the positive integer n.",
     "factors(12)", "[1, 2, 3, 4, 6, 12]",
     ["factors(12) == [1, 2, 3, 4, 6, 12]", "factors(1) == [1]", "factors(7) == [1, 7]",
      "factors(16) == [1, 2, 4, 8, 16]"],
     """
    return [d for d in range(1, n + 1) if n % d == 0]
"""),
    ("is_perfect_number", "n", "Return True if n equals the sum of its positive divisors excluding itself.",
     "is_perfect_number(28)", "True",
     ["is_perfect_number(28) == True", "is_perfect_number(6) == True", "is_perfect_number(12) == False",
      "is_perfect_number(1) == False"],
     """
    return n > 1 and sum(d for d in range(1, n) if n % d == 0) == n
"""),
    ("prime_factors", "n", "Return the prime factors of n (n >= 2) in increasing order, with repeats.",
     "prime_factors(60)", "[2, 2, 3, 5]",
     ["prime_factors(60) == [2, 2, 3, 5]", "prime_factors(2) == [2]", "prime_factors(13) == [13]",
      "prime_factors(100) == [2, 2, 5, 5]"],
     """
    out = []
    d = 2
    while n > 1:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    return out
"""),
    ("to_binary", "n",
     "Return the binary representation of the non-negative integer n as a string, without the '0b' prefix.",
     "to_binary(10)", "'1010'",
     ["to_binary(10) == '1010'", "to_binary(0) == '0'", "to_binary(1) == '1'", "to_binary(255) == '11111111'"],
     """
    return bin(n)[2:]
"""),
    ("from_binary", "s", "Convert a string of 0s and 1s to the integer it represents in binary.",
     'from_binary("1010")', "10",
     ["from_binary('1010') == 10", "from_binary('0') == 0", "from_binary('1') == 1",
      "from_binary('11111111') == 255"],
     """
    return int(s, 2)
"""),
    ("count_ones_binary", "n", "Return how many 1 bits are in the binary representation of the non-negative integer n.",
     "count_ones_binary(13)", "3",
     ["count_ones_binary(13) == 3", "count_ones_binary(0) == 0", "count_ones_binary(255) == 8",
      "count_ones_binary(8) == 1"],
     """
    return bin(n).count("1")
"""),
    ("to_roman", "n", "Convert an integer from 1 to 3999 to a Roman numeral.",
     "to_roman(1994)", "'MCMXCIV'",
     ["to_roman(1994) == 'MCMXCIV'", "to_roman(1) == 'I'", "to_roman(4) == 'IV'", "to_roman(9) == 'IX'",
      "to_roman(58) == 'LVIII'"],
     """
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    out = ""
    for v, s in zip(values, symbols):
        while n >= v:
            out += s
            n -= v
    return out
"""),
    ("from_roman", "s", "Convert a Roman numeral to an integer.",
     'from_roman("MCMXCIV")', "1994",
     ["from_roman('MCMXCIV') == 1994", "from_roman('III') == 3", "from_roman('IV') == 4",
      "from_roman('LVIII') == 58"],
     """
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and values[c] < values[s[i + 1]]:
            total -= values[c]
        else:
            total += values[c]
    return total
"""),
    ("clamp", "x, low, high", "Return x limited to the range [low, high].",
     "clamp(15, 0, 10)", "10",
     ["clamp(15, 0, 10) == 10", "clamp(-5, 0, 10) == 0", "clamp(5, 0, 10) == 5", "clamp(10, 0, 10) == 10"],
     """
    return max(low, min(x, high))
"""),
    ("sign", "x", "Return 1 if x is positive, -1 if x is negative, and 0 if x is zero.",
     "sign(-3.5)", "-1",
     ["sign(-3.5) == -1", "sign(0) == 0", "sign(7) == 1", "sign(-1) == -1"],
     """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
"""),
    ("grade", "score",
     "Return the letter grade for a score from 0 to 100: 'A' for 90 and above, 'B' for 80-89, "
     "'C' for 70-79, 'D' for 60-69 and 'F' below 60.",
     "grade(85)", "'B'",
     ["grade(85) == 'B'", "grade(90) == 'A'", "grade(100) == 'A'", "grade(59) == 'F'", "grade(60) == 'D'",
      "grade(79) == 'C'"],
     """
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
"""),
    ("fizzbuzz", "n",
     "Return a list of strings for 1 to n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
     "'FizzBuzz' for multiples of both, otherwise the number as a string.",
     "fizzbuzz(5)", "['1', '2', 'Fizz', '4', 'Buzz']",
     ["fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']", "fizzbuzz(15)[-1] == 'FizzBuzz'",
      "fizzbuzz(1) == ['1']", "len(fizzbuzz(30)) == 30", "fizzbuzz(6)[5] == 'Fizz'"],
     """
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
"""),
    ("fibonacci_list", "n", "Return a list of the first n Fibonacci numbers, starting with 0 and 1.",
     "fibonacci_list(6)", "[0, 1, 1, 2, 3, 5]",
     ["fibonacci_list(6) == [0, 1, 1, 2, 3, 5]", "fibonacci_list(0) == []", "fibonacci_list(1) == [0]",
      "fibonacci_list(2) == [0, 1]"],
     """
    out = []
    a, b = 0, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out
"""),
    ("multiplication_table", "n", "Return a list of the first 10 multiples of n: n, 2n, ..., 10n.",
     "multiplication_table(3)", "[3, 6, 9, 12, 15, 18, 21, 24, 27, 30]",
     ["multiplication_table(3) == [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]",
      "multiplication_table(0) == [0] * 10", "multiplication_table(1) == list(range(1, 11))"],
     """
    return [n * k for k in range(1, 11)]
"""),
    ("sum_multiples_3_or_5", "n", "Return the sum of all numbers below n that are multiples of 3 or 5.",
     "sum_multiples_3_or_5(10)", "23",
     ["sum_multiples_3_or_5(10) == 23",
      "sum_multiples_3_or_5(1) == 0",
      "sum_multiples_3_or_5(16) == 60"],
     """
    return sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)
"""),
    ("nth_prime", "n", "Return the nth prime number, where nth_prime(1) is 2.",
     "nth_prime(5)", "11",
     ["nth_prime(1) == 2",
      "nth_prime(5) == 11",
      "nth_prime(10) == 29"],
     """
    count = 0
    k = 1
    while count < n:
        k += 1
        if all(k % d for d in range(2, int(k ** 0.5) + 1)):
            count += 1
    return k
"""),
    ("primes_up_to", "n", "Return a list of all prime numbers less than or equal to n.",
     "primes_up_to(10)", "[2, 3, 5, 7]",
     ["primes_up_to(10) == [2, 3, 5, 7]",
      "primes_up_to(1) == []",
      "primes_up_to(2) == [2]"],
     """
    return [k for k in range(2, n + 1) if all(k % d for d in range(2, int(k ** 0.5) + 1))]
"""),
    ("safe_divide", "a, b", "Return a divided by b, or None if b is zero.",
     "safe_divide(10, 2)", "5.0",
     ["safe_divide(10, 2) == 5.0",
      "safe_divide(1, 0) is None",
      "safe_divide(7, 7) == 1.0"],
     """
    if b == 0:
        return None
    return a / b
"""),
    ("percent", "part, whole", "Return part as a percentage of whole, rounded to one decimal place.",
     "percent(1, 3)", "33.3",
     ["percent(1, 4) == 25.0",
      "percent(1, 3) == 33.3",
      "percent(0, 5) == 0.0"],
     """
    return round(part / whole * 100, 1)
"""),
    ("count_divisors", "n", "Return how many positive integers divide the positive integer n evenly.",
     "count_divisors(12)", "6",
     ["count_divisors(12) == 6",
      "count_divisors(1) == 1",
      "count_divisors(7) == 2"],
     """
    return sum(1 for d in range(1, n + 1) if n % d == 0)
"""),
    ("is_armstrong", "n", "Return True if n equals the sum of its digits, each raised to the power of the number of digits.",
     "is_armstrong(153)", "True",
     ["is_armstrong(153) == True",
      "is_armstrong(123) == False",
      "is_armstrong(9) == True"],
     """
    digits = str(n)
    return n == sum(int(d) ** len(digits) for d in digits)
"""),
    # --- lists ---
    ("is_sorted", "numbers", "Return True if the list is sorted in non-decreasing order.",
     "is_sorted([1, 2, 2, 5])", "True",
     ["is_sorted([1, 2, 2, 5]) == True", "is_sorted([3, 1]) == False", "is_sorted([]) == True",
      "is_sorted([5]) == True", "is_sorted([1, 3, 2]) == False"],
     """
    return all(numbers[i] <= numbers[i + 1] for i in range(len(numbers) - 1))
"""),
    ("second_largest", "numbers",
     "Return the second largest distinct value in the list. The list has at least two distinct values.",
     "second_largest([4, 1, 7, 7, 3])", "4",
     ["second_largest([4, 1, 7, 7, 3]) == 4", "second_largest([1, 2]) == 1", "second_largest([10, 10, 9]) == 9",
      "second_largest([-1, -5, -3]) == -3"],
     """
    return sorted(set(numbers))[-2]
"""),
    ("running_sum", "numbers", "Return a list where element i is the sum of numbers[0] through numbers[i].",
     "running_sum([1, 2, 3, 4])", "[1, 3, 6, 10]",
     ["running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]", "running_sum([]) == []", "running_sum([5]) == [5]",
      "running_sum([1, -1, 1]) == [1, 0, 1]"],
     """
    out = []
    total = 0
    for x in numbers:
        total += x
        out.append(total)
    return out
"""),
    ("cumulative_max", "numbers", "Return a list where element i is the largest of numbers[0] through numbers[i].",
     "cumulative_max([1, 3, 2, 5, 4])", "[1, 3, 3, 5, 5]",
     ["cumulative_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]", "cumulative_max([]) == []",
      "cumulative_max([-1, -2]) == [-1, -1]"],
     """
    out = []
    for x in numbers:
        out.append(x if not out else max(out[-1], x))
    return out
"""),
    ("flatten", "nested", "Flatten a list of lists into a single list, one level deep.",
     "flatten([[1, 2], [3], [4, 5]])", "[1, 2, 3, 4, 5]",
     ["flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]", "flatten([]) == []", "flatten([[], [1]]) == [1]",
      "flatten([['a'], ['b', 'c']]) == ['a', 'b', 'c']"],
     """
    return [x for sub in nested for x in sub]
"""),
    ("sum_nested", "nested",
     "Return the sum of all numbers in a list whose elements are numbers or lists of numbers (one level deep).",
     "sum_nested([1, [2, 3], 4])", "10",
     ["sum_nested([1, [2, 3], 4]) == 10", "sum_nested([]) == 0", "sum_nested([[1], [2]]) == 3",
      "sum_nested([5]) == 5"],
     """
    total = 0
    for x in nested:
        total += sum(x) if isinstance(x, list) else x
    return total
"""),
    ("chunk", "items, size",
     "Split items into consecutive chunks of length size; the last chunk may be shorter.",
     "chunk([1, 2, 3, 4, 5], 2)", "[[1, 2], [3, 4], [5]]",
     ["chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "chunk([], 3) == []",
      "chunk([1, 2, 3], 3) == [[1, 2, 3]]", "chunk([1, 2, 3], 1) == [[1], [2], [3]]"],
     """
    return [items[i:i + size] for i in range(0, len(items), size)]
"""),
    ("rotate_left", "items, k", "Rotate the list left by k positions.",
     "rotate_left([1, 2, 3, 4, 5], 2)", "[3, 4, 5, 1, 2]",
     ["rotate_left([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]", "rotate_left([], 3) == []",
      "rotate_left([1, 2, 3], 0) == [1, 2, 3]", "rotate_left([1, 2, 3], 3) == [1, 2, 3]",
      "rotate_left([1, 2, 3], 4) == [2, 3, 1]"],
     """
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]
"""),
    ("common_elements", "a, b",
     "Return the elements of a that also appear in b, in the order they appear in a, without duplicates.",
     "common_elements([1, 2, 3, 4], [4, 2, 6])", "[2, 4]",
     ["common_elements([1, 2, 3, 4], [4, 2, 6]) == [2, 4]", "common_elements([1, 1, 2], [1]) == [1]",
      "common_elements([], [1]) == []", "common_elements(['x', 'y'], ['z']) == []"],
     """
    out = []
    for x in a:
        if x in b and x not in out:
            out.append(x)
    return out
"""),
    ("index_of_max", "numbers",
     "Return the index of the largest number in the non-empty list. If it appears more than once, "
     "return the first index.",
     "index_of_max([3, 9, 2, 9])", "1",
     ["index_of_max([3, 9, 2, 9]) == 1", "index_of_max([1]) == 0", "index_of_max([-5, -1, -3]) == 1",
      "index_of_max([0, 0]) == 0"],
     """
    return numbers.index(max(numbers))
"""),
    ("count_positive", "numbers", "Return how many numbers in the list are greater than zero.",
     "count_positive([1, -2, 3, 0])", "2",
     ["count_positive([1, -2, 3, 0]) == 2", "count_positive([]) == 0", "count_positive([-1, -2]) == 0",
      "count_positive([5, 5]) == 2"],
     """
    return sum(1 for x in numbers if x > 0)
"""),
    ("count_in_range", "numbers, low, high", "Return how many numbers are between low and high, inclusive.",
     "count_in_range([1, 5, 10, 15], 5, 10)", "2",
     ["count_in_range([1, 5, 10, 15], 5, 10) == 2", "count_in_range([], 0, 1) == 0",
      "count_in_range([0, 1, 2], 1, 1) == 1"],
     """
    return sum(1 for x in numbers if low <= x <= high)
"""),
    ("remove_value", "items, value", "Return a new list with every occurrence of value removed.",
     "remove_value([1, 2, 1, 3], 1)", "[2, 3]",
     ["remove_value([1, 2, 1, 3], 1) == [2, 3]", "remove_value([], 1) == []", "remove_value([2], 1) == [2]",
      "remove_value(['a', 'a'], 'a') == []"],
     """
    return [x for x in items if x != value]
"""),
    ("remove_adjacent_duplicates", "items",
     "Return a list where each run of equal consecutive elements is reduced to a single element.",
     "remove_adjacent_duplicates([1, 1, 2, 2, 2, 1])", "[1, 2, 1]",
     ["remove_adjacent_duplicates([1, 1, 2, 2, 2, 1]) == [1, 2, 1]", "remove_adjacent_duplicates([]) == []",
      "remove_adjacent_duplicates([1]) == [1]", "remove_adjacent_duplicates(['a', 'a', 'b']) == ['a', 'b']"],
     """
    out = []
    for x in items:
        if not out or out[-1] != x:
            out.append(x)
    return out
"""),
    ("unique_sorted", "numbers", "Return the distinct values of the list in increasing order.",
     "unique_sorted([3, 1, 3, 2, 1])", "[1, 2, 3]",
     ["unique_sorted([3, 1, 3, 2, 1]) == [1, 2, 3]", "unique_sorted([]) == []", "unique_sorted([5, 5]) == [5]"],
     """
    return sorted(set(numbers))
"""),
    ("has_duplicates", "items", "Return True if any value appears more than once in the list.",
     "has_duplicates([1, 2, 3, 1])", "True",
     ["has_duplicates([1, 2, 3, 1]) == True", "has_duplicates([]) == False", "has_duplicates([1, 2]) == False",
      "has_duplicates(['a', 'a']) == True"],
     """
    return len(set(items)) != len(items)
"""),
    ("pairwise_differences", "numbers",
     "Return a list of the differences between consecutive numbers: numbers[i + 1] - numbers[i].",
     "pairwise_differences([1, 4, 9, 16])", "[3, 5, 7]",
     ["pairwise_differences([1, 4, 9, 16]) == [3, 5, 7]", "pairwise_differences([5]) == []",
      "pairwise_differences([]) == []", "pairwise_differences([3, 1]) == [-2]"],
     """
    return [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
"""),
    ("merge_sorted", "a, b", "Merge two sorted lists into one sorted list.",
     "merge_sorted([1, 4, 7], [2, 3, 8])", "[1, 2, 3, 4, 7, 8]",
     ["merge_sorted([1, 4, 7], [2, 3, 8]) == [1, 2, 3, 4, 7, 8]", "merge_sorted([], [1]) == [1]",
      "merge_sorted([1, 1], [1]) == [1, 1, 1]", "merge_sorted([], []) == []"],
     """
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]
"""),
    ("interleave", "a, b",
     "Return a list alternating elements of a and b, starting with a. Leftover elements of the longer "
     "list go at the end.",
     "interleave([1, 2, 3], ['a', 'b'])", "[1, 'a', 2, 'b', 3]",
     ["interleave([1, 2, 3], ['a', 'b']) == [1, 'a', 2, 'b', 3]", "interleave([], [1, 2]) == [1, 2]",
      "interleave([1], []) == [1]"],
     """
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out
"""),
    ("middle", "items",
     "Return the middle element of a list with an odd length, or the average of the two middle elements "
     "if the length is even.",
     "middle([1, 2, 3])", "2",
     ["middle([1, 2, 3]) == 2", "middle([1, 2, 3, 4]) == 2.5", "middle([7]) == 7", "middle([1, 3]) == 2"],
     """
    n = len(items)
    if n % 2 == 1:
        return items[n // 2]
    return (items[n // 2 - 1] + items[n // 2]) / 2
"""),
    ("product_list", "numbers", "Return the product of all numbers in the list; the product of an empty list is 1.",
     "product_list([2, 3, 4])", "24",
     ["product_list([2, 3, 4]) == 24", "product_list([]) == 1", "product_list([5]) == 5",
      "product_list([1, -2, 3]) == -6"],
     """
    result = 1
    for x in numbers:
        result *= x
    return result
"""),
    ("sum_even_numbers", "numbers", "Return the sum of the even numbers in the list.",
     "sum_even_numbers([1, 2, 3, 4])", "6",
     ["sum_even_numbers([1, 2, 3, 4]) == 6", "sum_even_numbers([]) == 0", "sum_even_numbers([1, 3]) == 0",
      "sum_even_numbers([-2, 2, 5]) == 0"],
     """
    return sum(x for x in numbers if x % 2 == 0)
"""),
    ("average_of_positives", "numbers",
     "Return the average of the positive numbers in the list, or 0 if there are none.",
     "average_of_positives([1, -1, 3])", "2.0",
     ["average_of_positives([1, -1, 3]) == 2", "average_of_positives([]) == 0",
      "average_of_positives([-1]) == 0", "average_of_positives([2, 4]) == 3"],
     """
    positives = [x for x in numbers if x > 0]
    if not positives:
        return 0
    return sum(positives) / len(positives)
"""),
    ("dot_product", "a, b", "Return the dot product of two lists of numbers with the same length.",
     "dot_product([1, 2, 3], [4, 5, 6])", "32",
     ["dot_product([1, 2, 3], [4, 5, 6]) == 32", "dot_product([], []) == 0", "dot_product([2], [3]) == 6",
      "dot_product([1, -1], [1, 1]) == 0"],
     """
    return sum(x * y for x, y in zip(a, b))
"""),
    ("count_pairs_sum", "numbers, target",
     "Return how many pairs of positions i < j have numbers[i] + numbers[j] == target.",
     "count_pairs_sum([1, 2, 3, 4], 5)", "2",
     ["count_pairs_sum([1, 2, 3, 4], 5) == 2", "count_pairs_sum([], 1) == 0",
      "count_pairs_sum([1, 1, 1], 2) == 3", "count_pairs_sum([2, 4], 5) == 0"],
     """
    count = 0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                count += 1
    return count
"""),
    ("binary_search", "numbers, target",
     "Return the index of target in the sorted list of distinct numbers, or -1 if it is not present.",
     "binary_search([1, 3, 5, 7, 9], 7)", "3",
     ["binary_search([1, 3, 5, 7, 9], 7) == 3", "binary_search([], 1) == -1", "binary_search([1, 3, 5], 4) == -1",
      "binary_search([2], 2) == 0", "binary_search([1, 2, 3, 4, 5, 6], 1) == 0"],
     """
    lo, hi = 0, len(numbers) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if numbers[mid] == target:
            return mid
        if numbers[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""),
    ("every_other", "items", "Return the elements at even indexes (0, 2, 4, ...).",
     "every_other([1, 2, 3, 4, 5])", "[1, 3, 5]",
     ["every_other([1, 2, 3, 4, 5]) == [1, 3, 5]",
      "every_other([]) == []",
      'every_other(["a", "b"]) == ["a"]'],
     """
    return items[::2]
"""),
    ("flatten_deep", "items", "Return a flat list of all non-list values in the arbitrarily nested list items, in order.",
     "flatten_deep([1, [2, [3, [4]]], 5])", "[1, 2, 3, 4, 5]",
     ["flatten_deep([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]",
      "flatten_deep([]) == []",
      "flatten_deep([[[]]]) == []"],
     """
    out = []
    for x in items:
        if isinstance(x, list):
            out.extend(flatten_deep(x))
        else:
            out.append(x)
    return out
"""),
    ("find_duplicates", "items", "Return the values that appear more than once, each listed once, in the order they first appear.",
     "find_duplicates([1, 2, 1, 3, 2, 1])", "[1, 2]",
     ["find_duplicates([1, 2, 1, 3, 2, 1]) == [1, 2]",
      "find_duplicates([1, 2, 3]) == []",
      'find_duplicates(["a", "a"]) == ["a"]'],
     """
    out = []
    for x in items:
        if items.count(x) > 1 and x not in out:
            out.append(x)
    return out
"""),
    ("difference", "a, b", "Return the elements of a that are not in b, keeping their order and repeats.",
     "difference([1, 2, 3, 4], [2, 4])", "[1, 3]",
     ["difference([1, 2, 3, 4], [2, 4]) == [1, 3]",
      "difference([1, 1, 2], [2]) == [1, 1]",
      "difference([], [1]) == []"],
     """
    return [x for x in a if x not in b]
"""),
    ("cumulative_product", "numbers", "Return a list where each element is the product of all elements up to and including that "
     "position.",
     "cumulative_product([1, 2, 3, 4])", "[1, 2, 6, 24]",
     ["cumulative_product([1, 2, 3, 4]) == [1, 2, 6, 24]",
      "cumulative_product([]) == []",
      "cumulative_product([3, 0, 5]) == [3, 0, 0]"],
     """
    out = []
    total = 1
    for x in numbers:
        total *= x
        out.append(total)
    return out
"""),
    ("moving_average", "numbers, k", "Return the average of every window of k consecutive numbers.",
     "moving_average([1, 2, 3, 4], 2)", "[1.5, 2.5, 3.5]",
     ["moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]",
      "moving_average([5, 5, 5], 3) == [5.0]",
      "moving_average([1, 2], 3) == []"],
     """
    return [sum(numbers[i:i + k]) / k for i in range(len(numbers) - k + 1)]
"""),
    ("normalize", "numbers", "Scale the numbers so the smallest becomes 0.0 and the largest becomes 1.0.",
     "normalize([0, 5, 10])", "[0.0, 0.5, 1.0]",
     ["normalize([0, 5, 10]) == [0.0, 0.5, 1.0]",
      "normalize([2, 4]) == [0.0, 1.0]",
      "normalize([4, 2, 3]) == [1.0, 0.0, 0.5]"],
     """
    lo, hi = min(numbers), max(numbers)
    return [(x - lo) / (hi - lo) for x in numbers]
"""),
    ("min_max", "numbers", "Return a tuple with the smallest and the largest number in the list.",
     "min_max([3, 1, 4, 1, 5])", "(1, 5)",
     ["min_max([3, 1, 4, 1, 5]) == (1, 5)",
      "min_max([7]) == (7, 7)",
      "min_max([-2, -8]) == (-8, -2)"],
     """
    return (min(numbers), max(numbers))
"""),
    ("median", "numbers", "Return the median of a non-empty list of numbers.",
     "median([4, 1, 3, 2])", "2.5",
     ["median([3, 1, 2]) == 2",
      "median([4, 1, 3, 2]) == 2.5",
      "median([5]) == 5"],
     """
    s = sorted(numbers)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
"""),
    ("mode", "numbers", "Return the most common number in the list. If there is a tie, return the smallest of them.",
     "mode([1, 2, 2, 3])", "2",
     ["mode([1, 2, 2, 3]) == 2",
      "mode([4, 4, 1, 1]) == 1",
      "mode([7]) == 7"],
     """
    return min(numbers, key=lambda x: (-numbers.count(x), x))
"""),
    ("matrix_sum", "matrix", "Return the sum of all numbers in the matrix, given as a list of rows.",
     "matrix_sum([[1, 2], [3, 4]])", "10",
     ["matrix_sum([[1, 2], [3, 4]]) == 10",
      "matrix_sum([]) == 0",
      "matrix_sum([[]]) == 0"],
     """
    return sum(sum(row) for row in matrix)
"""),
    ("bubble_sort", "numbers", "Return a new list with the numbers sorted in ascending order, using bubble sort.",
     "bubble_sort([3, 1, 2])", "[1, 2, 3]",
     ["bubble_sort([3, 1, 2]) == [1, 2, 3]",
      "bubble_sort([]) == []",
      "bubble_sort([5, -1, 5]) == [-1, 5, 5]"],
     """
    out = list(numbers)
    for i in range(len(out)):
        for j in range(len(out) - 1 - i):
            if out[j] > out[j + 1]:
                out[j], out[j + 1] = out[j + 1], out[j]
    return out
"""),
    ("remove_none", "items", "Return the list with every None removed. Other values, including 0 and False, stay.",
     "remove_none([1, None, 2, None])", "[1, 2]",
     ["remove_none([1, None, 2, None]) == [1, 2]",
      "remove_none([None]) == []",
      "remove_none([0, False]) == [0, False]"],
     """
    return [x for x in items if x is not None]
"""),
    ("sort_by_length", "words", "Return the words sorted from shortest to longest. Words of equal length keep their original "
     "order.",
     'sort_by_length(["ccc", "a", "bb"])', "['a', 'bb', 'ccc']",
     ['sort_by_length(["ccc", "a", "bb"]) == ["a", "bb", "ccc"]',
      'sort_by_length(["bb", "aa", "c"]) == ["c", "bb", "aa"]',
      "sort_by_length([]) == []"],
     """
    return sorted(words, key=len)
"""),
    ("sort_by_second", "pairs", "Return the list of tuples sorted by their second element.",
     "sort_by_second([(1, 3), (2, 1), (3, 2)])", "[(2, 1), (3, 2), (1, 3)]",
     ["sort_by_second([(1, 3), (2, 1), (3, 2)]) == [(2, 1), (3, 2), (1, 3)]",
      "sort_by_second([]) == []"],
     """
    return sorted(pairs, key=lambda p: p[1])
"""),
    ("group_by_first_letter", "words", "Return a dictionary mapping each first letter to the list of words that start with it, in their "
     "original order.",
     'group_by_first_letter(["apple", "avocado", "banana"])', "{'a': ['apple', 'avocado'], 'b': ['banana']}",
     ['group_by_first_letter(["apple", "avocado", "banana"]) == {"a": ["apple", "avocado"], "b": ["banana"]}',
      "group_by_first_letter([]) == {}"],
     """
    groups = {}
    for w in words:
        groups.setdefault(w[0], []).append(w)
    return groups
"""),
    ("split_pairs", "s", 'Split s into pieces of two characters. If the last piece has one character, add "_" to it.',
     'split_pairs("abc")', "['ab', 'c_']",
     ['split_pairs("abcdef") == ["ab", "cd", "ef"]',
      'split_pairs("abc") == ["ab", "c_"]',
      'split_pairs("") == []'],
     """
    if len(s) % 2:
        s += "_"
    return [s[i:i + 2] for i in range(0, len(s), 2)]
"""),
    ("parse_ints", "s", "Return a list of the integers in the comma-separated string s. Spaces around numbers are "
     "allowed.",
     'parse_ints("1,2,3")', "[1, 2, 3]",
     ['parse_ints("1,2,3") == [1, 2, 3]',
      'parse_ints("10") == [10]',
      'parse_ints(" 4, 5 ") == [4, 5]',
      'parse_ints("") == []'],
     """
    return [int(p) for p in s.split(",") if p.strip()]
"""),
    ("vowels_per_word", "sentence", "Return a list with the number of vowels (a, e, i, o, u) in each word of the lowercase sentence.",
     'vowels_per_word("hello world")', "[2, 1]",
     ['vowels_per_word("hello world") == [2, 1]',
      'vowels_per_word("sky") == [0]',
      'vowels_per_word("") == []'],
     """
    return [sum(1 for c in w if c in "aeiou") for w in sentence.split()]
"""),
    # --- matrices ---
    ("transpose", "matrix", "Return the transpose of a rectangular matrix given as a list of rows.",
     "transpose([[1, 2, 3], [4, 5, 6]])", "[[1, 4], [2, 5], [3, 6]]",
     ["transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "transpose([[1]]) == [[1]]",
      "transpose([[1, 2]]) == [[1], [2]]"],
     """
    return [list(row) for row in zip(*matrix)]
"""),
    ("diagonal_sum", "matrix", "Return the sum of the main diagonal of a square matrix.",
     "diagonal_sum([[1, 2], [3, 4]])", "5",
     ["diagonal_sum([[1, 2], [3, 4]]) == 5", "diagonal_sum([[5]]) == 5",
      "diagonal_sum([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6"],
     """
    return sum(matrix[i][i] for i in range(len(matrix)))
"""),
    ("matrix_max", "matrix", "Return the largest value in a non-empty matrix given as a list of rows.",
     "matrix_max([[1, 5], [3, 2]])", "5",
     ["matrix_max([[1, 5], [3, 2]]) == 5", "matrix_max([[-1]]) == -1", "matrix_max([[1, 2], [9, 0]]) == 9"],
     """
    return max(max(row) for row in matrix)
"""),
    ("row_sums", "matrix", "Return a list with the sum of each row of the matrix.",
     "row_sums([[1, 2], [3, 4]])", "[3, 7]",
     ["row_sums([[1, 2], [3, 4]]) == [3, 7]", "row_sums([]) == []", "row_sums([[5]]) == [5]"],
     """
    return [sum(row) for row in matrix]
"""),
    ("column_sums", "matrix", "Return a list with the sum of each column of a non-empty rectangular matrix.",
     "column_sums([[1, 2], [3, 4]])", "[4, 6]",
     ["column_sums([[1, 2], [3, 4]]) == [4, 6]", "column_sums([[1, 2, 3]]) == [1, 2, 3]",
      "column_sums([[1], [2], [3]]) == [6]"],
     """
    return [sum(col) for col in zip(*matrix)]
"""),
    ("identity_matrix", "n", "Return the n by n identity matrix as a list of lists.",
     "identity_matrix(2)", "[[1, 0], [0, 1]]",
     ["identity_matrix(2) == [[1, 0], [0, 1]]", "identity_matrix(1) == [[1]]", "identity_matrix(0) == []",
      "identity_matrix(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]"],
     """
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
"""),
    # --- dictionaries ---
    ("char_frequency", "s", "Return a dictionary mapping each character in s to the number of times it appears.",
     'char_frequency("hello")', "{'h': 1, 'e': 1, 'l': 2, 'o': 1}",
     ["char_frequency('hello') == {'h': 1, 'e': 1, 'l': 2, 'o': 1}", "char_frequency('') == {}",
      "char_frequency('aab') == {'a': 2, 'b': 1}"],
     """
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return counts
"""),
    ("word_count", "sentence",
     "Return a dictionary mapping each word of the sentence, lowercased, to how many times it appears.",
     'word_count("the cat and the hat")', "{'the': 2, 'cat': 1, 'and': 1, 'hat': 1}",
     ["word_count('the cat and the hat') == {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}", "word_count('') == {}",
      "word_count('A a') == {'a': 2}"],
     """
    counts = {}
    for w in sentence.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return counts
"""),
    ("group_by_length", "words",
     "Return a dictionary mapping each word length to the list of words with that length, in their "
     "original order.",
     'group_by_length(["a", "bb", "c", "dd"])', "{1: ['a', 'c'], 2: ['bb', 'dd']}",
     ["group_by_length(['a', 'bb', 'c', 'dd']) == {1: ['a', 'c'], 2: ['bb', 'dd']}", "group_by_length([]) == {}",
      "group_by_length(['hi']) == {2: ['hi']}"],
     """
    groups = {}
    for w in words:
        groups.setdefault(len(w), []).append(w)
    return groups
"""),
    ("invert_dict", "d", "Return a new dictionary with the keys and values of d swapped. The values of d are unique.",
     "invert_dict({'a': 1, 'b': 2})", "{1: 'a', 2: 'b'}",
     ["invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}", "invert_dict({}) == {}",
      "invert_dict({1: 'x'}) == {'x': 1}"],
     """
    return {v: k for k, v in d.items()}
"""),
    ("merge_dicts", "a, b",
     "Return a new dictionary with the entries of a and b. When a key is in both, add the two values.",
     "merge_dicts({'x': 1, 'y': 2}, {'y': 3, 'z': 4})", "{'x': 1, 'y': 5, 'z': 4}",
     ["merge_dicts({'x': 1, 'y': 2}, {'y': 3, 'z': 4}) == {'x': 1, 'y': 5, 'z': 4}", "merge_dicts({}, {}) == {}",
      "merge_dicts({'a': 1}, {}) == {'a': 1}"],
     """
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out
"""),
    ("key_with_max_value", "d",
     "Return the key of the non-empty dictionary d that has the largest value. On a tie, return the key "
     "that comes first in d.",
     "key_with_max_value({'a': 3, 'b': 7, 'c': 5})", "'b'",
     ["key_with_max_value({'a': 3, 'b': 7, 'c': 5}) == 'b'", "key_with_max_value({'x': 1}) == 'x'",
      "key_with_max_value({'a': 2, 'b': 2}) == 'a'"],
     """
    best = None
    for k, v in d.items():
        if best is None or v > d[best]:
            best = k
    return best
"""),
    ("sum_dict_values", "d", "Return the sum of all values in the dictionary d.",
     "sum_dict_values({'a': 1, 'b': 2})", "3",
     ["sum_dict_values({'a': 1, 'b': 2}) == 3", "sum_dict_values({}) == 0", "sum_dict_values({'x': -4}) == -4"],
     """
    return sum(d.values())
"""),
    ("filter_dict", "d, threshold",
     "Return a new dictionary with only the entries of d whose value is greater than threshold.",
     "filter_dict({'a': 1, 'b': 5, 'c': 3}, 2)", "{'b': 5, 'c': 3}",
     ["filter_dict({'a': 1, 'b': 5, 'c': 3}, 2) == {'b': 5, 'c': 3}", "filter_dict({}, 0) == {}",
      "filter_dict({'a': 1}, 1) == {}"],
     """
    return {k: v for k, v in d.items() if v > threshold}
"""),
    ("zip_to_dict", "keys, values", "Return a dictionary pairing each key with the value at the same position.",
     "zip_to_dict(['a', 'b'], [1, 2])", "{'a': 1, 'b': 2}",
     ["zip_to_dict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}", "zip_to_dict([], []) == {}",
      "zip_to_dict([1], ['x']) == {1: 'x'}"],
     """
    return dict(zip(keys, values))
"""),
    # --- small algorithms ---
    ("balanced_brackets", "s", "Return True if the brackets (), [] and {} in s are balanced and correctly nested.",
     'balanced_brackets("([]{})")', "True",
     ['balanced_brackets("([]{})") == True',
      'balanced_brackets("([)]") == False',
      'balanced_brackets("") == True',
      'balanced_brackets("((") == False'],
     """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack
"""),
    ("max_subarray_sum", "numbers", "Return the largest sum of any non-empty run of consecutive numbers in the list.",
     "max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])", "6",
     ["max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6",
      "max_subarray_sum([-3, -1]) == -1",
      "max_subarray_sum([5]) == 5"],
     """
    best = cur = numbers[0]
    for x in numbers[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
"""),
    ("two_sum", "numbers, target", "Return a tuple (i, j) with i < j of the first pair of indexes whose numbers add up to target, "
     "checking i in order and then j. Return None if there is no such pair.",
     "two_sum([2, 7, 11, 15], 9)", "(0, 1)",
     ["two_sum([2, 7, 11, 15], 9) == (0, 1)",
      "two_sum([1, 2, 3], 7) is None",
      "two_sum([3, 3], 6) == (0, 1)",
      "two_sum([1, 4, 2, 3], 5) == (0, 1)"],
     """
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return (i, j)
    return None
"""),
    ("count_islands_1d", "cells", "Return how many separate groups of consecutive 1s appear in the list of 0s and 1s.",
     "count_islands_1d([1, 1, 0, 1, 0, 0, 1, 1])", "3",
     ["count_islands_1d([1, 1, 0, 1, 0, 0, 1, 1]) == 3",
      "count_islands_1d([0, 0]) == 0",
      "count_islands_1d([1]) == 1",
      "count_islands_1d([]) == 0"],
     """
    count = 0
    prev = 0
    for c in cells:
        if c == 1 and prev == 0:
            count += 1
        prev = c
    return count
"""),
    ("longest_run", "items", "Return the length of the longest run of equal consecutive elements.",
     "longest_run([1, 1, 2, 2, 2, 3])", "3",
     ["longest_run([1, 1, 2, 2, 2, 3]) == 3",
      "longest_run([]) == 0",
      'longest_run(["a"]) == 1',
      "longest_run([4, 5, 6]) == 1"],
     """
    best = cur = 0
    prev = object()
    for x in items:
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    return best
"""),
    ("pascal_row", "n", "Return row n of Pascal's triangle as a list, where row 0 is [1].",
     "pascal_row(4)", "[1, 4, 6, 4, 1]",
     ["pascal_row(0) == [1]",
      "pascal_row(1) == [1, 1]",
      "pascal_row(4) == [1, 4, 6, 4, 1]"],
     """
    row = [1]
    for _ in range(n):
        row = [1] + [a + b for a, b in zip(row, row[1:])] + [1]
    return row
"""),
    ("matrix_multiply", "a, b", "Return the product of the matrices a and b, each given as a list of rows.",
     "matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])", "[[19, 22], [43, 50]]",
     ["matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]",
      "matrix_multiply([[2]], [[3]]) == [[6]]",
      "matrix_multiply([[1, 2, 3]], [[1], [2], [3]]) == [[14]]"],
     """
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]
"""),
    ("spiral_order", "matrix", "Return the elements of the matrix in clockwise spiral order, starting at the top-left.",
     "spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])", "[1, 2, 3, 6, 9, 8, 7, 4, 5]",
     ["spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]",
      "spiral_order([[1, 2], [3, 4]]) == [1, 2, 4, 3]",
      "spiral_order([]) == []"],
     """
    out = []
    m = [list(row) for row in matrix]
    while m:
        out.extend(m.pop(0))
        m = [list(row) for row in zip(*m)][::-1]
    return out
"""),
    ("word_wrap", "text, width", "Split text into lines of whole words, each line at most width characters long. Return the list "
     "of lines.",
     'word_wrap("the quick brown fox", 10)', "['the quick', 'brown fox']",
     ['word_wrap("the quick brown fox", 10) == ["the quick", "brown fox"]',
      'word_wrap("hello", 10) == ["hello"]',
      'word_wrap("a b c", 1) == ["a", "b", "c"]',
      'word_wrap("", 5) == []'],
     """
    lines = []
    line = ""
    for w in text.split():
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= width:
            line += " " + w
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines
"""),
    ("longest_common_prefix", "words",
     "Return the longest string that every word in the list starts with. Return '' if there is none.",
     'longest_common_prefix(["flower", "flow", "flight"])', "'fl'",
     ["longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'", "longest_common_prefix(['dog', 'car']) == ''",
      "longest_common_prefix(['same', 'same']) == 'same'", "longest_common_prefix(['a']) == 'a'",
      "longest_common_prefix([]) == ''"],
     """
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
    return prefix
"""),
    ("missing_number", "numbers",
     "The list holds every integer from 0 to n except one, in any order. Return the missing integer.",
     "missing_number([3, 0, 1])", "2",
     ["missing_number([3, 0, 1]) == 2", "missing_number([0]) == 1", "missing_number([1]) == 0",
      "missing_number([0, 1, 2, 4, 5]) == 3"],
     """
    n = len(numbers)
    return n * (n + 1) // 2 - sum(numbers)
"""),
    ("is_isogram", "word", "Return True if no letter appears more than once in the word, ignoring case.",
     'is_isogram("lumber")', "True",
     ["is_isogram('lumber') == True", "is_isogram('letter') == False", "is_isogram('') == True",
      "is_isogram('Alpha') == False"],
     """
    letters = word.lower()
    return len(set(letters)) == len(letters)
"""),
    ("even_odd_split", "numbers",
     "Return a tuple of two lists: the even numbers and the odd numbers, each in their original order.",
     "even_odd_split([1, 2, 3, 4, 5])", "([2, 4], [1, 3, 5])",
     ["even_odd_split([1, 2, 3, 4, 5]) == ([2, 4], [1, 3, 5])", "even_odd_split([]) == ([], [])",
      "even_odd_split([7]) == ([], [7])", "even_odd_split([0, -2]) == ([0, -2], [])"],
     """
    evens = [x for x in numbers if x % 2 == 0]
    odds = [x for x in numbers if x % 2 != 0]
    return (evens, odds)
"""),
    ("is_arithmetic", "numbers",
     "Return True if the difference between every pair of consecutive numbers is the same. "
     "Lists with fewer than two numbers count as arithmetic.",
     "is_arithmetic([3, 5, 7, 9])", "True",
     ["is_arithmetic([3, 5, 7, 9]) == True", "is_arithmetic([1, 2, 4]) == False", "is_arithmetic([5]) == True",
      "is_arithmetic([]) == True", "is_arithmetic([10, 7, 4]) == True"],
     """
    if len(numbers) < 2:
        return True
    step = numbers[1] - numbers[0]
    return all(numbers[i + 1] - numbers[i] == step for i in range(len(numbers) - 1))
"""),
    ("next_prime", "n", "Return the smallest prime number that is greater than n.",
     "next_prime(14)", "17",
     ["next_prime(14) == 17", "next_prime(0) == 2", "next_prime(2) == 3", "next_prime(7) == 11",
      "next_prime(100) == 101"],
     """
    k = n + 1
    while k < 2 or any(k % d == 0 for d in range(2, int(k ** 0.5) + 1)):
        k += 1
    return k
"""),
    ("to_base", "n, base",
     "Return the non-negative integer n written in the given base (2 to 16) as a string, using "
     "uppercase letters A-F for digits above 9.",
     "to_base(255, 16)", "'FF'",
     ["to_base(255, 16) == 'FF'", "to_base(10, 2) == '1010'", "to_base(0, 8) == '0'", "to_base(64, 8) == '100'",
      "to_base(35, 16) == '23'"],
     """
    digits = "0123456789ABCDEF"
    if n == 0:
        return "0"
    out = ""
    while n > 0:
        out = digits[n % base] + out
        n //= base
    return out
"""),
    ("remove_duplicate_chars", "s",
     "Return s with only the first occurrence of each character kept, in their original order.",
     'remove_duplicate_chars("programming")', "'progamin'",
     ["remove_duplicate_chars('programming') == 'progamin'", "remove_duplicate_chars('') == ''",
      "remove_duplicate_chars('aaa') == 'a'", "remove_duplicate_chars('abcabc') == 'abc'"],
     """
    out = ""
    for c in s:
        if c not in out:
            out += c
    return out
"""),
    ("count_words_starting_with", "sentence, letter",
     "Return how many words in the sentence start with the given letter, ignoring case.",
     'count_words_starting_with("She sells sea shells", "s")', "4",
     ["count_words_starting_with('She sells sea shells', 's') == 4",
      "count_words_starting_with('apple banana avocado', 'A') == 2",
      "count_words_starting_with('', 'x') == 0", "count_words_starting_with('hello world', 'z') == 0"],
     """
    return sum(1 for w in sentence.split() if w[0].lower() == letter.lower())
"""),
    ("nth_largest", "numbers, n",
     "Return the nth largest number in the list, where n = 1 is the largest. Repeated values count "
     "separately.",
     "nth_largest([4, 1, 7, 3], 2)", "4",
     ["nth_largest([4, 1, 7, 3], 2) == 4", "nth_largest([5], 1) == 5", "nth_largest([3, 3, 1], 2) == 3",
      "nth_largest([9, 2, 6], 3) == 2"],
     """
    return sorted(numbers, reverse=True)[n - 1]
"""),
    ("max_depth", "items",
     "Return how deeply lists are nested inside the list items. A list with no inner lists has depth 1.",
     "max_depth([1, [2, [3]], 4])", "3",
     ["max_depth([1, [2, [3]], 4]) == 3", "max_depth([]) == 1", "max_depth([1, 2]) == 1",
      "max_depth([[], [[]]]) == 3"],
     """
    deepest = 0
    for x in items:
        if isinstance(x, list):
            deepest = max(deepest, max_depth(x))
    return deepest + 1
"""),
    ("alternating_sum", "numbers",
     "Return numbers[0] - numbers[1] + numbers[2] - numbers[3] + ..., alternating between adding "
     "and subtracting.",
     "alternating_sum([10, 3, 5, 2])", "10",
     ["alternating_sum([10, 3, 5, 2]) == 10", "alternating_sum([]) == 0", "alternating_sum([4]) == 4",
      "alternating_sum([1, 1, 1]) == 1"],
     """
    return sum(x if i % 2 == 0 else -x for i, x in enumerate(numbers))
"""),
]


def mid_prompt(name: str, params: str, doc: str, call: str, result: str) -> str:
    doc_lines = textwrap.wrap(doc, width=76)
    body = "\n    ".join(doc_lines)
    return f'def {name}({params}):\n    """{body}\n\n    >>> {call}\n    {result}\n    """'


def mid_task_items() -> list[tuple[str, str, str]]:
    """(name, prompt, tests) tuples in the same shape as the smoke tasks."""
    out = []
    for name, params, doc, call, result, asserts, _ in MID_TASKS:
        tests = "".join(f"assert {a}\n" for a in asserts)
        out.append((name, mid_prompt(name, params, doc, call, result), tests))
    return out


def check_references() -> None:
    names = [t[0] for t in MID_TASKS]
    assert len(names) == len(set(names)), "duplicate task names"
    failures = []
    for (name, prompt, tests), task in zip(mid_task_items(), MID_TASKS):
        call, result, ref = task[3], task[4], task[6]
        program = prompt + "\n" + ref.lstrip("\n") + "\n" + tests + f"assert {call} == {result}\n"
        try:
            exec(compile(program, name, "exec"), {})
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {type(e).__name__}: {e}")
    if failures:
        raise SystemExit("reference failures:\n  " + "\n  ".join(failures))
    print(f"all {len(MID_TASKS)} reference solutions pass their tests and examples")


if __name__ == "__main__":
    check_references()
