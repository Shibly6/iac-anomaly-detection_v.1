resource "aws_s3_bucket" "public_via_policy" {
  bucket = "public-via-policy-bucket"
}

resource "aws_s3_bucket_acl" "public_via_policy_acl" {
  bucket = aws_s3_bucket.public_via_policy.id
  acl    = "private"  # Deceptive! Policy makes it public.
}

resource "aws_s3_bucket_policy" "allow_public_access" {
  bucket = aws_s3_bucket.public_via_policy.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = "*",
      Action    = "s3:GetObject",
      Resource  = "${aws_s3_bucket.public_via_policy.arn}/*"
    }]
  })
}