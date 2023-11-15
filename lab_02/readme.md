In this lab I tried to describe the genome of the population by encoding all possible combinations of states of the game Nim.
In this way I realized it is impossible to use this encoding when the number of rows increases.

My main objective was to try to let the algorithm find an optimal solution by having no idea of the possible strategies. It should only have a detailed description of the game he is playing.

I noticed that each possible move can be described with the effect it produces and I was able to distiguish a total of four categories:
a. a move that leaves in the row 0 objects
b. a move that leaves in the row 1 oject
c. a move that leaves in the row an even number of objects
d. a move that leaves in the row an odd number of objects

I also thought it was possible to assign a category to each row by looking at the possible categories of moves applicable to that row:
A. a row in which they are only possible type a moves
B. a row in which they are only possible type a and b moves
C. a row in which they are only possible type a, b and c moves
D. a row in which they are possible all types of moves (a, b, c, d)

By using the categories of rows, I decided to assign a category to the actual state too, assigning the same category of the highest category of row that is in that state. For example, the state {1, 3, 5} is a D state because the categories of its rows are, in order, {A, C, D}; the state {1, 2, 2, 1, 2} is a category B state.

In this way I can describe any possible state using a string of 5 charachters:
1. string[0] is the category of the state: [A, B, C, D]
2. string[1] is the number of rows of type A: [Z, O, P, D] = [Zero, One, Pari (even), Dispari (odd)]
3. string[2] is the number of rows of type B: [Z, O, P, D] = [Zero, One, Pari (even), Dispari (odd)]
4. string[3] is the number of rows of type C: [Z, O, P, D] = [Zero, One, Pari (even), Dispari (odd)]
5. string[4] is the number of rows of type D: [Z, O, P, D] = [Zero, One, Pari (even), Dispari (odd)]

For example, the state {1, 3, 5, 7} has, as row types, {A, C, D, D}. Its encoding is DOZPP; the state {2, 1, 3, 2} = {B, A, C, B}, has the encoding COPDZ

In the same way, I encode all possible moves describing the type of row where it is applied, and the type of move using a tuple.
move = (type_of_row | type_of_move)
In this way it is possible to describe all possible moves, for all possible states, for all possible rows, in 10 categories:
A. (A | a) -> on type A rows you can only use type a moves
B. (B | a), (B | b) -> on type B rows you can use type a and type b moves 
C. (C | a), (C | b), (C | c) -> on type C rows you can use type a, type b and type c moves
D. (D | a), (D | b), (D | c), (D | d) -> on type D rows you can use type a, type b, type c and type d moves

The idea of the genome of a strategy is based on the usage of an hash_map that has as keys the encoding of the state ("XXXXX") and as value a list of size 10 where they are stored the weights of each type of move.
The weights are randomly initialized with a number between 1 and 10 and, when playing, if a type of move is not allowed in that state (i.e. in a state of type B you can't use type C and type D moves) its value is set to 0.
In this way, the value corresponding to the encoding of the state, describes the weights of all allowed types of moves.
When a type of move is chosen based on the highest value in the list, a random move of the same type is picked from the possible moves of that state.

This idea is based on the fact that it is not important which move you choose, but the resulting state in which you arrive.

After the generation of the initial population, it is calculated the score of each strategy and they are chosen the strongest 25% of the population.
This strongest quarter is used to generate the new strategies by using crossover and mutation functions.
In the crossover function it is chosen, for each of the 10 weights, in each value of the hash_map, one from the mother and one from the father, with the exception that, if one, between the mother and the father, has 0 in a specific index of the list, that 0 value is propagated to the child because it tells that that type of move is illegal in that state and it is useless to update it.
In the mutation function 30% of genes are modified using a gaussian distribution with mean = 0 and standard_deviation = 0.6

Thanks to my colleague Filippo Greco (S309529), I used a corrected version of the optimal agent that uses the nim_sum as fitness function.
This agent always wins if it starts as first player for each number of starting rows, except for 1, 4, 8, 12, etc. for each value multiple of 4.
The objective of my agent is to understand this optimal strategy without any pre-computed fitness function and it should increase its ability by executing the genetic algorithm.

To date, 15/11/2023, this algorithm, is not working at all. I tried to do my best but I'm pretty sure the ecoding applied in the genome is missing something.
The only notewothy result can be seen when playing against the "original version" of the optimal algorithm that had a bug and basically played in a random way. Against it, my strategy obtained a percentage of winnign close to 80% both playing first or second.

The two files lab_02_version_9 in this folder are my final work. They are the same but they calculate the best strategy by playing first in one case and second in the other case. This was made just for speed-up the process because I noticed that these programs are executed in single core and cannot be parallelized. The only way to execute them faster is to manually generate different data on different programs that will be executed on different CPU cores.
In the test_version folder there are all my tries, starting from the earlier versions in which I haven't already thought about this way of encoding.
In the aknowledgement folder there are some programs I thought are interesting and some papers that I used after understanding my code was not efficient to try to find the bug. They suggested me my entire code is a bug and I should review it from the beginning. I also discovered (again, thanks Filippo Greco) that there is a theorem (Sprague-Grundy theorem) that describes all nim-like games that are under the family of impartial games. I was surprised to notice that, even if not completely correct, my argument about categorizing moves, states, and rows, wasn't too far from the real correct description of this type of game.

Because of the deadline I have to consider this commit as my final work for the lab_02 but I will continue working on this because I already spent too much hours in this and I can't give up now, I'm too proud of my work.