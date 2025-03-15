# **********************************************************************
# Run the API to Access the Fitted Model ----
#
# Purpose ----
# To run the API that enables users/validators to access the model
# through an interface.
# **********************************************************************

# Install and Load the Required Packages ----
## plumber ----
if (require("plumber")) {
  require("plumber")
} else {
  install.packages("plumber", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Process a Plumber API ----
# This allows us to process a plumber API
api <- plumber::plumb("./Subsystem3-Crop-Production/API.R")

# Run the API on a specific port ----
# Specify a constant localhost port to use
api$run(host = "127.0.0.1", port = 5022)
