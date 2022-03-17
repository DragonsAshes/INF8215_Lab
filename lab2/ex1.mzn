% Retault Virgile : 2164296
% Foucher Sebastien : 2162248

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

% output : [14, 12, 17, 10, 27]