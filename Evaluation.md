
# Evaluation

Let \{$T_1^{(1)}, ...,T_M^{(1)}$\},...,\{$T_1^{(N)}, ...,T_M^{(N)}$\} be our ground truth, 
\{$P_1^{(1)}, ...,P_M^{(1)}$\},...,\{$P_1^{(N)}, ...,P_M^{(N)}$\} the prediction.

on diagonal we count the events of truth matching prediction

$\sum_{i=1}^N |\{T_1^{(i)}, ...,T_M^{(i)}\}\cap \{P_1^{(i)}, ...,P_M^{(i)}\}|$

so for each diagonal entry we count over all data points (N) and all predictions (M) the total amount when truth is matching prediction.

$C_{kk} =\sum_{i=1}^N \sum_{m=1}^M \chi{1}\{T_m^{(i)}=k,P_m^{(i)}=k\} $

in contrast on off-diagonal we count all events where truth was not covered by prediction.
For the off-diagonal we take only the difference sets into account. If truth for data point i was for example {fish, curry, rice-dish} and prediction was {fish, curry, soup }, we only count 
$T^i$ = {rice_dish} and $P^i$ = {soup} for the off-daigonal. Formally we do the following transformation:

\begin{align}
\{ \tilde{T_m}^{(i)} \}_{m=1,..,M1}  
\end{align}


$\setminus \{P_m^{(i)}\}_{m=1,..M}$


$\\{ \tilde{P_m}^{(i)} \\}_{m=1,..,M2} $ =

$\\{ P_m^{(i)}\\}_{m=1,..,M}\setminus $ 
 
$\\{T_m^{(i)}\\}_{m=1,..,M}$

Eventually we count all events of the ground truth that were not captured in predictions, which results in

$C_{kl} =\sum_{i=1}^N \sum_{m1=1}^{M1} \sum_{m2=1}^{M2}
\frac{1}{M1}\chi{1}\{\tilde{T}_{m1}^{(i)}=k,\tilde{P}_{m2}^{(i)}=l\} $

Once we have computed a multi-class - multi-label confusion matrix the classification report is straight forward.

Precision for class k is computed as

$Pr_k=\frac{C_{kk}}{\sum_j C_{kj}}$

and Recall

$R_k=\frac{C_{kk}}{\sum_j C_{jk}}$


