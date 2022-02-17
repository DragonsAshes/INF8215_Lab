%-2p Quoridor tournament-

include "globals.mzn";

int: MAX_T = 30;%nombre maximum de creneaux
int: G = 3;%nombre de tables de tournoi
int: P = 13;%nombre de joueurs

var 1..MAX_T: T;%temps a minimiser

array[1..MAX_T,1..G,1..2] of var 0..P: x;

% <A COMPLETER>
predicate all_different_exclude_zero(array[int] of var int: a) =
  forall (i, j in index_set(a) where i<j) (
    a[i] != 0 /\ a[j] !=0 -> a[i]!=a[j]
  );

int: nb_match = 78; %nombre de matchs = 2 parmis P

% Regle sur le joueur 1
array[int] of int: l = ([a | a in 1..MAX_T where (a mod 3) != 0 ]);

constraint forall(p in 2..P) (
    x[l[p-1],1,1] = 1 /\ x[l[p-1], 1, 2] = p
);


% Chaque joueur affronte tout le monde
constraint forall([(x[t, g, 1] < x[t, g, 2] /\ x[t,g,1]!=0) \/ (x[t,g,1]=0 /\ x[t,g,2]=0) | g in 1..G, t in 1..T]);
constraint sum([1 |  g in 1..G, t in 1..T where x[t,g,1]!=0])=nb_match;
constraint forall([x[t,g,1] = 0 /\ x[t, g, 2] = 0 | g in 1..G, t in T..MAX_T]);
constraint all_different_exclude_zero([x[t, g, 1]+100*x[t,g,2] | g in 1..G, t in 1..MAX_T]);
constraint forall(t in 1..MAX_T) (
    forall(p in 1..P) (
      sum([1 | g in 1..G where x[t,g,1] = p \/ x[t,g,2] = p]) <= 1
    )
);

% Chaque joueur ne doit pas jouer 3 parties consécutives
constraint forall(t in 1..MAX_T-2) (
    forall(p in 1..P) (
      sum([1 | g in 1..G where x[t,g,1] = p \/ x[t,g,2] = p]) + sum([1 | g in 1..G where x[t+1,g,1] = p \/ x[t+1,g,2] = p]) + sum([1 | g in 1..G where x[t+2,g,1] = p \/ x[t+2,g,2] = p]) <= 2
    )
);

solve minimize T; 
output
["Schedule ("++show(T)++" time slots)\n"]
++
[
    if j = 1 then "\n" else " " endif ++ show(x[i,j,1])++" "++show(x[i,j,2])++ if j = G then " " else " -" endif
      | i in 1..MAX_T, j in 1..G
]