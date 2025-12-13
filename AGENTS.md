## Site Snapshot
- Personal portfolio and blog for Vijay Kethanaboyina, built with Jekyll and the dark-poole theme, currently served at `https://vkethana.com`.
- Homepage `index.md` pulls in project highlights via `_includes/projectlist.html`, a bio, and outbound links to the blog feed, GitHub, and LinkedIn.
- Long-form posts live in `_posts/`, while standalone project pages (for coursework, research, and apps) are Markdown/HTML files at the repo root; shared styling resides in `assets/css/styles.scss` and `_sass/main.scss`, with media under `assets/images/`.
- Plugins configured in `_config.yml` (`jekyll-seo-tag`, `jekyll-feed`, `jekyll-sitemap`) handle metadata, RSS, and sitemap generation.

## Development Inputs
- Ensure a Ruby + Bundler toolchain (vendor bundle targets Ruby 3.4.0); install deps with `bundle install` and run locally with `bundle exec jekyll serve` or the `build_site.sh` helper.
- Keep `_config.yml` metadata (`title`, `description`, `author`, social handles) up to date and provide front matter for any new pages or posts.
- Source material needed for new content: project write-ups, blog drafts, hero images, and navigation updates before editing the Markdown sources or HTML includes.
- Confirm deployment workflow (e.g., GitHub Pages vs. custom hosting) and any required build flags/env vars before shipping changes.

