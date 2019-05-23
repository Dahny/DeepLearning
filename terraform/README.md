# HOWTO

First, install terraform 

Then, create an SSH key and add it to your project according to here https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys

Create and get a service account key (https://cloud.google.com/iam/docs/creating-managing-service-account-keys) and place it in the terraform folder as account.json

Spin up an instance using `terraform apply -var="name=*name associated with ssh key*"`

This will automatically prepare the benchmarking environment. Once it is done, you can connect to the instance using ssh and do whatever you want.

Once you're done, don't forget to run `terraform destroy` to prevent leaving the instance running when it is not in use.