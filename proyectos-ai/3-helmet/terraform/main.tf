# =============================================================================
# Helmet API — ECS Task Definition (TEMPLATE / PLACEHOLDER)
# =============================================================================
# Este archivo es un template documentado para desplegar la Helmet API en AWS ECS.
# NO es funcional sin adaptar: requires real ECR image, VPC, subnets, SG, etc.
# Usado como referencia de infraestructura reproducible en el bootcamp.
# =============================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------- ECR Repository (placeholder) ----------

resource "aws_ecr_repository" "helmet_api" {
  name                 = "helmet-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project   = "helmet"
    ManagedBy = "terraform"
  }
}

# ---------- ECS Cluster ----------

resource "aws_ecs_cluster" "helmet" {
  name = "helmet-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Project   = "helmet"
    ManagedBy = "terraform"
  }
}

# ---------- Task Definition ----------

resource "aws_ecs_task_definition" "helmet_api" {
  family                   = "helmet-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "helmet-api"
      image     = "${var.ecr_repository}:latest"
      cpu       = var.cpu
      memory    = var.memory
      essential = true

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "MODEL_NAME", value = "llama3.2" },
        { name = "OLLAMA_ENDPOINT", value = "" },
      ]

      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:helmet/api-key"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/helmet-api"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Project   = "helmet"
    ManagedBy = "terraform"
  }
}

# ---------- IAM Roles (minimal) ----------

data "aws_caller_identity" "current" {}

resource "aws_iam_role" "ecs_execution" {
  name = "helmet-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "helmet-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_ssm" {
  name = "helmet-ssm-read"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameters", "secretsmanager:GetSecretValue"]
      Resource = "*"
    }]
  })
}

# ---------- CloudWatch Log Group ----------

resource "aws_cloudwatch_log_group" "helmet" {
  name              = "/ecs/helmet-api"
  retention_in_days = 14

  tags = {
    Project   = "helmet"
    ManagedBy = "terraform"
  }
}
