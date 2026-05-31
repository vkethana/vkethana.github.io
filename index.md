---
layout: base
---

<div class="home-title-block" aria-labelledby="home-title">
  <div class="title-row">
    <span class="title-ornament" aria-hidden="true">§</span>
    <h1 id="home-title" class="home-title">vijay kethanaboyina</h1>
    <span class="title-ornament" aria-hidden="true">§</span>
  </div>
</div>

<section class="home-section" aria-labelledby="recent-title">
  <h2 id="recent-title" class="visually-hidden">Posts</h2>
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
