# **********************************************************************
# The API to Access the Fitted Model ----
#
# Purpose ----
# To enable users/validators to access the model through an interface.
# **********************************************************************

# Install and Load the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## plumber ----
if (require("plumber")) {
  require("plumber")
} else {
  install.packages("plumber", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# The saved model can then be loaded later as follows:
model_glmStepAIC <- readRDS("../Subsystem3-Crop-Production/1.Curve-Fitting/Models/model_glmStepAIC.rds")
# print(model_glmStepAIC)


#* @apiTitle Difference Level in Crop Production for a Competitive Economy

#* @apiDescription Used to predict the "difference level" (more than just the standard deviation) in crop production between farmers. Low values are correlated with a competitive economy.

#* @param r_var_x_0 Average monthly revenue generated per category of crop producer from crop production.
#* @param water_energy_util_var_x_1 The utility derived from water and energy
#* @param dem_var_x_2
#* @param supp_var_x_3
#* @param diff_sup_dem_var_x_4
#* @param tal_var_x_5
#* @param vert_s_x_6
#* @param exp_var_x_7
#* @param mal_var_x_8

#* @get /welfare

predict_difference_level <-
  function(r_var_x_0, water_energy_util_var_x_1, dem_var_x_2, supp_var_x_3,
           diff_sup_dem_var_x_4,
           tal_var_x_5, vert_s_x_6, exp_var_x_7, mal_var_x_8) {
    # Create a data frame using the arguments
    to_be_predicted <-
      data.frame(
        r_var = as.numeric(r_var_x_0),
        water_energy_util_var = as.numeric(water_energy_util_var_x_1),
        dem_var = as.numeric(dem_var_x_2),
        supp_var = as.numeric(supp_var_x_3),
        diff_sup_dem_var = as.numeric(diff_sup_dem_var_x_4),
        tal_var = as.numeric(tal_var_x_5),
        vert_s = as.numeric(vert_s_x_6),
        exp_var = as.numeric(exp_var_x_7),
        mal_var = as.numeric(mal_var_x_8)
      )
    # Make a prediction based on the data frame
    predict(model_glmStepAIC, to_be_predicted)
  }

# Minimum point from Particle Swarm Optimization:
# curl -X GET "http://127.0.0.1:5022/welfare?r_var_x_0=0.00506700&water_energy_util_var_x_1=12786.45306018&dem_var_x_2=0.00138888&supp_var_x_3=0.00027777&diff_sup_dem_var_x_4=3558784.65671254&tal_var_x_5=0.00022606&vert_s_x_6=2082.59651300&exp_var_x_7=0.01922163&mal_var_x_8=0.01933329" -H  "accept: */*"

# Random point resulting in a prediction of 474.2293:
# curl -X GET "http://127.0.0.1:5022/welfare?r_var_x_0=506.6906911&water_energy_util_var_x_1=1061.929187&dem_var_x_2=0.001114205&supp_var_x_3=0.001290323&diff_sup_dem_var_x_4=6.246897115&tal_var_x_5=0.008657511&vert_s_x_6=720&exp_var_x_7=0.750750751&mal_var_x_8=1.208150219" -H "accept: */*"