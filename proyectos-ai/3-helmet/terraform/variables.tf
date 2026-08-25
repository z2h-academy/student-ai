variable "aws_region" {
  description = "Region de AWS para desplegar Helmet API"
  type        = string
  default     = "us-east-1"
}

variable "ecr_repository" {
  description = "URI del repositorio ECR (sin tag)"
  type        = string
  default     = "helmet-api"
}

variable "cpu" {
  description = "CPU units para la task (1 vCPU = 1024)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Memoria en MB para la task"
  type        = number
  default     = 1024
}
