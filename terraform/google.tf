provider "google" {
  credentials = "${file("account.json")}"
  project     = "buoyant-planet-125720"
  region      = "europe-west4"
}

variable "name" {
  type = "string"
}

data "google_compute_image" "my_image" {
  family  = "pytorch-1-1-gpu"
  project = "ml-images"
}

resource "google_compute_instance" "default" {
  name         = "deeplearnz"
  machine_type = "custom-8-30720"
  zone         = "europe-west4-a"

  boot_disk {
    initialize_params {
      image = "${data.google_compute_image.my_image.self_link}",
      size = 100
    }
  }


  network_interface {
    network = "default"
    access_config {
        // Ephemeral IP - leaving this block empty will generate a new external IP and assign it to the machine
    }
  }

  guest_accelerator {
    type = "nvidia-tesla-p100",
    count = 1
  }

  scheduling{
    on_host_maintenance = "TERMINATE" // Need to terminate GPU on maintenance
  }

  service_account {
    scopes = ["userinfo-email", "compute-ro", "storage-ro"]
  }



  # Copy the whole project to the CLAUWD
  provisioner "file" {
    connection {
      type     = "ssh"
      user     = "${var.name}"
    }
    source = "../../DeepLearning"
    destination = "~/"
  }

  # Run the prepare script
  provisioner "remote-exec" {
    connection {
      type     = "ssh"
      user     = "${var.name}"
    }
    inline = [
      "cd ~/DeepLearning/Benchmark",
      "sudo chmod +x ./prepare.sh",
      "./prepare.sh"
    ]
  }

  #metadata_startup_script = "${file("../Benchmark/prepare.sh")}"
}

output "ip" {
 value = "${google_compute_instance.default.network_interface.0.access_config.0.nat_ip}"
}