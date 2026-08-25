output "api_url" {
  description = "URL del ALB/ECS service (placeholder — reemplazar con endpoint real)"
  value       = "https://<ALB_DNS_NAME>/api/ask"
}

output "task_arn" {
  description = "ARN de la ECS task definition"
  value       = aws_ecs_task_definition.helmet_api.arn
}
