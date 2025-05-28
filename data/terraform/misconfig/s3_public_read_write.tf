resource "aws_s3_bucket" "public_read_write" {
  bucket = "public-read-write-bucket"
}

resource "aws_s3_bucket_acl" "public_read_write_acl" {
  bucket = aws_s3_bucket.public_read_write.id
  acl    = "public-read-write"
}