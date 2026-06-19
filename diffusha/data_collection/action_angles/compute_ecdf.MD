"""
action - 
feed joystick **id** samples, see for each datapoint what ecdf
value it sits within the id array (so always - id array + length 1 operation?), and then
create an array of paired values (1 - ecdf(reconstruction_error), 0)

do the same for joystick **ood** samples, do the same against id array
create an array of paired values (1 - ecdf(reconstruction_error), 1)

**speculation** is that both will stay pretty high in terms of anomaly score, 
but the magnitude and freq of joystick ood will be bigger

and i do the above for
1. the raw action values
2. the angles
3. the magnitude of actions

**NOTE** PRIORITY
1. angles
2. magnitude of actions
3. raw action values

**TODO** breakdown
1. for raw action values
    - implement within `joystick_diffdagger.py` file for computing ecdf, write to csv (label, score)
    - post-hoc analysis: run roc_auc_curves function + visualization code after having collected all csv data

2/3. for angles and magnitude of actions
    - get trajectory angles of id trajectories and ood trajectories beforehand
    - implement within `joystick_diffdagger.py` file for computing ecdf, write to csv (label, score)
    - post-hoc analysis: run roc_auc_curves function + visualiaztion code after having collected all csv data
    

**NOTE** so what am i trying to get out of this?
- seeing if angle is a good signal to be used as the OOD signal (would be binary? 0 or 1)
 rather than the entire, raw user action signal, and so that i can use this for
 another 0 or 1 signal (the state ood) to determine final OOD-ness
- or I can use the ecdf value of this to do CDF * CDF for gamma decision making
(similar to how I did earlier)
"""