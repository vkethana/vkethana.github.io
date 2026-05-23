---
layout: base
---

<section class="home-hero" aria-labelledby="home-title">
  <div class="hero-art">
    <img class="hero-portrait" src="/assets/images/portrait.jpeg" alt="Vijay Kethanaboyina">
  </div>

  <div class="hero-copy">
    <h1 id="home-title">Vijay Kethanaboyina</h1>
    <p class="hero-subtitle">computer science at UC Berkeley</p>
    <p class="hero-summary">
      I work on machine learning systems, NLP, computer vision, and language-learning tools.
      I also write about projects, research ideas, and things I am trying to understand.
    </p>
    <div class="social-row" aria-label="Social links">
      <a href="https://www.github.com/vkethana" aria-label="GitHub">
        <svg viewBox="0 0 24 24" role="img" aria-hidden="true">
          <path d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.6v-2.2c-3.3.7-4-1.4-4-1.4-.5-1.2-1.2-1.6-1.2-1.6-1-.7.1-.7.1-.7 1.1.1 1.7 1.2 1.7 1.2 1 .1.7 2.1 3.4 1.5.1-.7.4-1.2.7-1.5-2.6-.3-5.3-1.3-5.3-5.8 0-1.3.5-2.3 1.2-3.2-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.8.1 3.1.8.9 1.2 1.9 1.2 3.2 0 4.5-2.7 5.5-5.3 5.8.4.4.8 1.1.8 2.2v3.2c0 .4.2.7.8.6A12 12 0 0 0 12 .5Z"/>
        </svg>
      </a>
      <a href="https://www.linkedin.com/in/vkethana/" aria-label="LinkedIn">
        <span aria-hidden="true">in</span>
      </a>
      <a href="{{ "/feed.xml" | prepend: site.baseurl }}" aria-label="RSS feed">
        <span aria-hidden="true">rss</span>
      </a>
    </div>
  </div>
</section>

<section class="home-section" aria-labelledby="recent-title">
  <div class="section-heading">
    <h2 id="recent-title">Recent</h2>
    <a href="{{ "/vjposts" | prepend: site.baseurl }}">All posts</a>
  </div>

  <div class="recent-list">
    {% assign recent_posts = site.posts | where_exp: "post", "post.published != false" | slice: 0, 8 %}
    {% for post in recent_posts %}
      {% assign words = post.content | strip_html | number_of_words %}
      {% assign minutes = words | divided_by: 200 %}
      {% if minutes < 1 %}{% assign minutes = 1 %}{% endif %}
      <article class="recent-item">
        <a class="recent-thumb" href="{{ post.url | prepend: site.baseurl }}" aria-label="{{ post.title }}">
          {% if post.featured_image %}
            <img src="{{ post.featured_image }}" alt="">
          {% else %}
            <span>{{ post.title | slice: 0 }}</span>
          {% endif %}
        </a>
        <div class="recent-body">
          <h3><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h3>
          <p>
            <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%-d %B %Y" }}</time>
            <span>{{ words }} words</span>
            <span>{{ minutes }} min</span>
          </p>
        </div>
      </article>
    {% endfor %}
  </div>
</section>

<section class="home-section compact-bio" aria-labelledby="about-title">
  <h2 id="about-title">About</h2>
  <p>
    I am pursuing a bachelor's degree in computer science at UC Berkeley.
    Coursework includes operating systems, machine learning, deep learning, deep reinforcement learning,
    computer vision, data structures, algorithms, FPGA design, historical linguistics, and Sanskrit.
  </p>
</section>
