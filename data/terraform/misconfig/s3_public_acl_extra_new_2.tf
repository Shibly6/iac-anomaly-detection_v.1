resource "aws_s3_bucket" "anonymous_read_access" {
  bucket = "anonymous-read-access-bucket"
}

resource "aws_s3_bucket_acl" "anonymous_read_access_acl" {
  bucket = aws_s3_bucket.anonymous_read_access.id
  acl    = "public-read"
}