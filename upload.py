repo_url = "https://huggingface.co/mltrev23/codet5-small-code-summarization-ruby"

from huggingface_hub import Repository

repo = Repository(local_dir="checkpoint", # note that this directory must not exist already
                  clone_from=repo_url,
                  git_user="mltrev23",
                  git_email="trevor.dev23@gmail.com",
                  use_auth_token=True,
)
model.save_pretrained("/content/checkpoint")
tokenizer.save_pretrained("/content/checkpoint")

repo.push_to_hub(commit_message="First commit")