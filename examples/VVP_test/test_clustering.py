import easyvvuq as uq
import os
from vvp import ensemble_vvp
import pandas as pd
import numpy as np

#Load the average force output vector for each threshold value
def load_forces(dirname, **kwargs):
    threshold = float(dirname.split('_')[-1])
    df = pd.read_csv(dirname + '/average_forces.csv')

    return [threshold, df]

#The VVP agregation_function, compares the average force output vector for each threshold value
# with the reference values
def check_convergence(forces_thresholds, **kwargs):
    ## sorting by increasing threshold value
    tmp = sorted(forces_thresholds,key=lambda x: (x[0],x[1]))
    tmp.reverse()
    forces_thresholds = tmp
    
    thresholds = [forces[0] for forces in forces_thresholds]
    min_threshold = min(thresholds)
    print('The reference is set to the output obtained with threshold value = %.10f' % min_threshold)

    for forces in forces_thresholds:
        if forces[0] == min(thresholds):
            ref_forces = forces[1]['resulting_force'].to_numpy()

    for forces in forces_thresholds:
        tsh_forces = forces[1]['resulting_force'].to_numpy()
        print('Threshold value = %.8f' % forces[0])
        print('Forces norm with current threshold = %.3f' % np.linalg.norm(tsh_forces),
                  ', exact = %.6f' % np.linalg.norm(ref_forces),
                  ', error = %.6f' % np.linalg.norm((tsh_forces-ref_forces)),
                  ', relative error = %.6f' % np.linalg.norm((tsh_forces-ref_forces)/ref_forces))
        print('=========================================================')

# Run an ensemble of simulations for each threshold value contained in 'clustering_threshold_list'
# and compute the average global force for each ensemble
def run_campaign(clustering_threshold_list, restart=False):

  if restart is False:
      # Set up a fresh campaign called "coffee_pce"
      my_campaign = uq.Campaign(name='scema_clustering_threshold_')
      # Define parameter space
      params = {
          "clustering_threshold": {"type": "float", "min": 0.0, "max": 1000.0, "default": 10e-6}
      }
      # Create an encoder, decoder and collater for PCE test app
      encoder = uq.encoders.GenericEncoder(
          template_fname='inputs_dogbone.template',
          delimiter='$',
          target_filename='inputs_dogbone.json')
      decoder = uq.decoders.SimpleCSV(target_filename="./macroscale_log/loadedbc_force.csv",
                                      output_columns=["timestep", "resulting_force"],
                                      header=0)
      collater = uq.collate.AggregateSamples(average=False)
      # Add the app (automatically set as current app)
      my_campaign.add_app(name="scema",
                          params=params,
                          encoder=encoder,
                          decoder=decoder,
                          collater=collater)
      # Create the sampler
      vary = {
          "clustering_threshold": clustering_threshold_list
      }
      my_sampler = uq.sampling.BasicSweep(vary)
      my_campaign.set_sampler(my_sampler)
      my_campaign.draw_samples(num_samples=len(clustering_threshold_list), replicas=2)
      my_campaign.populate_runs_dir()

      cwd = "/home/uccamva/tmp/vvp_clustering_threshold" #os.getcwd()
      cmd = "mpiexec {}/dealammps inputs_dogbone.json > out.log".format(cwd)
      my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(cmd))
      my_campaign.collate()

      my_campaign.save_state('campaign_state.json')

  else:
      my_campaign = uq.Campaign(state_file="campaign_state.json", work_dir="./")

  results_path=my_campaign.campaign_dir+"/results/"
  os.mkdir(results_path)

  results_data_frame = my_campaign.get_collation_result()
  results_data_frame.to_csv(results_path+"results.csv")
  grouped_results = results_data_frame.groupby(['clustering_threshold','timestep']).mean()
  grouped_results.to_csv(results_path+"group_results.csv")
  grouped_results.reset_index(level=[1,0], inplace=True)

  for threshold in clustering_threshold_list:
      avg_results_path=results_path+"threshold_{}/".format(threshold)
      os.mkdir(avg_results_path)
      threshold_grouped_results = grouped_results[grouped_results["clustering_threshold"]==threshold]
      threshold_grouped_results[['timestep','resulting_force']].to_csv(avg_results_path+"average_forces.csv", index=False)

  return results_path


if __name__ == "__main__":

  clustering_thresholds=[10**i for i in range(-8,4,1)]
  results_dir = run_campaign(clustering_thresholds, restart=True)

  ensemble_vvp(results_dir, load_forces, check_convergence)
