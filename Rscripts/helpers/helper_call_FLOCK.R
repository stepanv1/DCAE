#call ecexutable of FLOCK

helper_call_Flock <- function(data, FLOCKDIR = '/home/sgrinek/bin/'){

  old_dir= getwd()
  setwd(FLOCKDIR)

  # save external data file
  write.table(data, file = "FLOCK_data_file.txt", quote = FALSE, sep = "\t", row.names = FALSE)
    
  cmd <- "./flock2 ./FLOCK_data_file.txt"
  system(cmd)

  # read results from external results file
  out <- read.table("flock_results.txt", header = TRUE, sep = "\t")
  
  setwd(old_dir)
  
  return(out)
}

    