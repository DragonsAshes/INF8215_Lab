%-Carres-

include "globals.mzn";

array[1..5] of var 10..100: x;

% constraint <A COMPLETER>;
constraint pow(x[5],2) = pow(x[1],2) +pow(x[2],2) + pow(x[3],2)+ pow(x[4],2);
constraint alldifferent(x);
% solve <A COMPLETER>;
solve minimize x[5];

output [
  show(x)
];