# correct/s3_private_restrictive_policy.tf
resource "aws_s3_bucket" "private_with_policy" {
  bucket = "private-bucket-with-policy"
}

resource "aws_s3_bucket_acl" "private_with_policy_acl" {
  bucket = aws_s3_bucket.private_with_policy.id
  acl    = "private"
}

resource "aws_s3_bucket_policy" "deny_public_access" {
  bucket = aws_s3_bucket.private_with_policy.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Deny",
      Principal = "*",
      Action    = "s3:GetObject",
      Condition = { Bool = { "aws:SecureTransport" = "false" } },
      Resource  = "${aws_s3_bucket.private_with_policy.arn}/*"
    }]
  })
}