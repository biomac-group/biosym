*** This data was generated with AddBiomechanics (www.addbiomechanics.org) ***
AddBiomechanics was written by Keenon Werling.

Automatic processing achieved the following marker errors (averaged
over all frames of all trials):

- Avg. Marker RMSE      = 3.55 cm
- Avg. Max Marker Error = 6.88 cm

WARNING! Dynamics fitting was skipped for the following reason:

  Force plates had zero force data across all time steps.

The following trials were processed to perform automatic body scaling
and marker registration:

trial: Subject3_trial1
  - Avg. Marker RMSE      = 3.54 cm
  - Avg. Marker Max Error = 6.92 cm
  - WARNING: 8 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 10 marker(s)!
  --> See IK/Subject3_trial1_ik_summary.txt for more details.

trial: Subject3_trial2
  - Avg. Marker RMSE      = 3.56 cm
  - Avg. Marker Max Error = 6.75 cm
  - WARNING: 9 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 3 marker(s)!
  --> See IK/Subject3_trial2_ik_summary.txt for more details.

trial: Subject3_trial3
  - Avg. Marker RMSE      = 3.52 cm
  - Avg. Marker Max Error = 6.90 cm
  - WARNING: 6 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 3 marker(s)!
  --> See IK/Subject3_trial3_ik_summary.txt for more details.

trial: Subject3_trial4
  - Avg. Marker RMSE      = 3.64 cm
  - Avg. Marker Max Error = 7.02 cm
  - WARNING: 10 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 2 marker(s)!
  --> See IK/Subject3_trial4_ik_summary.txt for more details.

trial: Subject3_trial5
  - Avg. Marker RMSE      = 3.61 cm
  - Avg. Marker Max Error = 7.03 cm
  - WARNING: 8 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 7 marker(s)!
  --> See IK/Subject3_trial5_ik_summary.txt for more details.

trial: Subject3_trial6
  - Avg. Marker RMSE      = 3.45 cm
  - Avg. Marker Max Error = 6.66 cm
  - WARNING: 8 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 3 marker(s)!
  --> See IK/Subject3_trial6_ik_summary.txt for more details.


The model file containing optimal body scaling and marker offsets is:

Models/final.osim

This tool works by finding optimal scale factors and marker offsets at
the same time. If specified, it also runs a second optimization to
find mass parameters to fit the model dynamics to the ground reaction
force data.

Since you did not choose to run the dynamics fitting step,
Models/optimized_scale_and_markers.osim contains the same model as
Models/final.osim.

If you want to manually edit the marker offsets, you can modify the
<MarkerSet> in "Models/unscaled_but_with_optimized_markers.osim" (by
default this file contains the marker offsets found by the optimizer).
If you want to tweak the Scaling, you can edit
"Models/rescaling_setup.xml". If you change either of these files,
then run (FROM THE "Models" FOLDER, and not including the leading ">
"):

 > opensim-cmd run-tool rescaling_setup.xml
           # This will re-generate Models/optimized_scale_and_markers.osim


You do not need to re-run Inverse Kinematics unless you change
scaling, because the output motion files are already generated for you
as "*_ik.mot" files for each trial, but you are welcome to confirm our
results using OpenSim. To re-run Inverse Kinematics with OpenSim, to
verify the results of AddBiomechanics, you can use the automatically
generated XML configuration files. Here are the command-line commands
you can run (FROM THE "IK" FOLDER, and not including the leading "> ")
to verify IK results for each trial:

 > opensim-cmd run-tool Subject3_trial1_ik_setup.xml
           # This will create a results file IK/Subject3_trial1_ik_by_opensim.mot
 > opensim-cmd run-tool Subject3_trial2_ik_setup.xml
           # This will create a results file IK/Subject3_trial2_ik_by_opensim.mot
 > opensim-cmd run-tool Subject3_trial3_ik_setup.xml
           # This will create a results file IK/Subject3_trial3_ik_by_opensim.mot
 > opensim-cmd run-tool Subject3_trial4_ik_setup.xml
           # This will create a results file IK/Subject3_trial4_ik_by_opensim.mot
 > opensim-cmd run-tool Subject3_trial5_ik_setup.xml
           # This will create a results file IK/Subject3_trial5_ik_by_opensim.mot
 > opensim-cmd run-tool Subject3_trial6_ik_setup.xml
           # This will create a results file IK/Subject3_trial6_ik_by_opensim.mot


To run Inverse Dynamics with OpenSim, you can also use automatically
generated XML configuration files. WARNING: This AddBiomechanics run
did not attempt to fit dynamics (you need to have GRF data and enable
physics fitting in the web app), so the residuals will not be small
and YOU SHOULD NOT EXPECT THEM TO BE. That being said, to run inverse
dynamics the following commands should work (FROM THE "ID" FOLDER, and
not including the leading "> "):

 > opensim-cmd run-tool Subject3_trial1_id_setup.xml
           # This will create a results file ID/Subject3_trial1_id.sto
 > opensim-cmd run-tool Subject3_trial2_id_setup.xml
           # This will create a results file ID/Subject3_trial2_id.sto
 > opensim-cmd run-tool Subject3_trial3_id_setup.xml
           # This will create a results file ID/Subject3_trial3_id.sto
 > opensim-cmd run-tool Subject3_trial4_id_setup.xml
           # This will create a results file ID/Subject3_trial4_id.sto
 > opensim-cmd run-tool Subject3_trial5_id_setup.xml
           # This will create a results file ID/Subject3_trial5_id.sto
 > opensim-cmd run-tool Subject3_trial6_id_setup.xml
           # This will create a results file ID/Subject3_trial6_id.sto


The original unscaled model file is present in:

Models/unscaled_generic.osim

There is also an unscaled model, with markers moved to spots found by
this tool, at:

Models/unscaled_but_with_optimized_markers.osim

If you encounter errors, please submit a post to the AddBiomechanics
user forum on SimTK.org:

   https://simtk.org/projects/addbiomechanics