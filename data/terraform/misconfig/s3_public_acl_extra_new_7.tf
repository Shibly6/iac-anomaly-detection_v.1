resource "aws_s3_bucket" "public_website_content" {
  bucket = "public-website-content-bucket"
}

resource "aws_s3_bucket_acl" "public_website_content_acl" {
  bucket = aws_s3_bucket.public_website_content.id
  acl    = "public-read"
}