"""
Author: Christos Faloutsos
Date: May 2020
Goal:
	string editing distance, with variable-length don't care
	Useful for the AHT project with Jeremy, Catalina ++
"""

import argparse
import numpy as NP

STAR = "*"  # variable-length don't care character


def check_string(s):
	assert len(s) == 0 or s.isalpha(), "string should be alpha: " + s


def check_pattern(p):
	assert len(p) == 0 or p.isalpha() or (STAR in p), "illegal pattern: " + p


float_costs = False
if float_costs:
	CI = 2.0  # insertion cost (insertion into the pattern)
	CD = 1.0  # deletion cost (from the pattern)
	CS = 0.5  # substitution for non-* characters
	CSS = 0.1  # substitution for * character
	CSI = 0.2  # insertion for '*' - ie, gobbling up the string-character
	CSD = 0.0  # cost to ignore '*' - ie., match it with null string

# default, integer-valued, costs
CI = 20
CD = 10
CS = 5
CSS = 1
CSI = 2
CSD = 0

# typical values, with zero cost for '*'-matching
typical = False
if typical:
	CI = 1.0
	CD = 1.0
	CS = 1.0
	CSS = 0.0
	CSI = CSS
	CSD = 0.0


def string_edit_plus(p, s, verbose=0):
	if verbose > 0:
		print("sep: pattern = '{}', string = '{}'".format(p, s))
	m = len(p)
	n = len(s)

	# check_pattern(p)
	# check_string(s)

	# distance = 0
	# if m == n and m == 0:
	#     distance = 0

	dm = NP.empty(shape=(m + 1, n + 1), dtype=object)  # distance matrix

	# initialize string vs empty-pattern
	for j in range(n + 1):
		dm[0][j] = j * CI

	# initialize pattern vs empty-string
	assert dm[0][0] == 0.0, "[0,0] should be 0.0"
	for i in range(1, m + 1):  # pattern
		if STAR == p[i - 1]:  # cost of '*' matching 'null'
			dm[i][0] = dm[i - 1][0] + CSD
		else:
			dm[i][0] = dm[i - 1][0] + CD

	# do the iterations, by column
	# (ideally, we would like to do recursion + memoization...)

	for j in range(1, n + 1):
		for i in range(1, m + 1):
			# minus1 because we 'prepended' a null character
			# to pattern and string (sigh!)
			pc = p[i - 1]  # character of the pattern
			sc = s[j - 1]  # character of the string
			if STAR == pc:
				cs_star = CSS + dm[i - 1, j - 1]
				ci_star = CSI + dm[i, j - 1]
				cd_star = CSD + dm[i - 1, j]
				cost = min(cs_star, ci_star, cd_star)
			else:
				if pc == sc:
					cost = dm[i - 1, j - 1]
				else:
					cs = CS + dm[i - 1, j - 1]
					cd = CD + dm[i - 1, j]
					ci = CI + dm[i, j - 1]
					cost = min(cs, cd, ci)
			dm[i, j] = cost
	if verbose > 0:
		print(dm)
	distance = dm[m, n]
	return distance


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="calculate string-editing distance")
	parser.add_argument("-v", "--verbose", action="count", default=0)
	parser.add_argument("pattern", type=str, help="the pattern")
	parser.add_argument("string", type=str, help="the string")
	args = parser.parse_args()

	verbose = int(args.verbose)
	pattern = args.pattern
	string = args.string
	distance = 0

	check_pattern(pattern)
	check_string(string)

	distance = string_edit_plus(pattern, string, verbose=verbose)

	if verbose > 0:
		print("distance('{}', '{}') = {}; verbose = {}".format(pattern, string, distance, verbose))
	else:
		print("distance('{}', '{}')  = {}".format(pattern, string, distance))
