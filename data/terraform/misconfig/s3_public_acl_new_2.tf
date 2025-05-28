resource "aws_s3_bucket" "world_readable_new" {
  bucket = "world-readable-new-bucket"
}

resource "aws_s3_bucket_acl" "world_readable_acl_new" {
  bucket = aws_s3_bucket.world_readable_new.id
  acl    = "public-read-write"
}