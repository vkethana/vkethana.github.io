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
    {% assign cognateful_post = site.posts | where: "url", "/cognateful/" | first %}
    {% if cognateful_post %}
      {% assign words = cognateful_post.content | strip_html | number_of_words %}
      {% assign minutes = words | divided_by: 200 %}
      {% if minutes < 1 %}{% assign minutes = 1 %}{% endif %}
      <article class="recent-item">
        <a class="recent-thumb" href="{{ cognateful_post.url | prepend: site.baseurl }}" aria-label="{{ cognateful_post.title }}">
          {% if cognateful_post.featured_image %}
            <img src="{{ cognateful_post.featured_image }}" alt="">
          {% else %}
            <span>{{ cognateful_post.title | slice: 0 }}</span>
          {% endif %}
        </a>
        <div class="recent-body">
          <h3><a href="{{ cognateful_post.url | prepend: site.baseurl }}">{{ cognateful_post.title }}</a></h3>
          <p>
            <time datetime="{{ cognateful_post.date | date_to_xmlschema }}">{{ cognateful_post.date | date: "%-d %B %Y" }}</time>
            <span>{{ words }} words</span>
            <span>{{ minutes }} min</span>
          </p>
        </div>
      </article>
    {% endif %}

    {% assign recent_posts = site.posts | where_exp: "post", "post.published != false" %}
    {% for post in recent_posts %}
      {% if forloop.index == 2 %}
        {% assign mse_post = site.posts | where: "url", "/pusht/" | first %}
        {% if mse_post %}
          {% assign words = mse_post.content | strip_html | number_of_words %}
          {% assign minutes = words | divided_by: 200 %}
          {% if minutes < 1 %}{% assign minutes = 1 %}{% endif %}
          <article class="recent-item">
            <a class="recent-thumb" href="{{ mse_post.url | prepend: site.baseurl }}" aria-label="{{ mse_post.title }}">
              {% if mse_post.featured_image %}
                <img src="{{ mse_post.featured_image }}" alt="">
              {% else %}
                <span>{{ mse_post.title | slice: 0 }}</span>
              {% endif %}
            </a>
            <div class="recent-body">
              <h3><a href="{{ mse_post.url | prepend: site.baseurl }}">Training an MSE Policy via Imitation Learning</a></h3>
              <p>
                <time datetime="{{ mse_post.date | date_to_xmlschema }}">{{ mse_post.date | date: "%-d %B %Y" }}</time>
                <span>{{ words }} words</span>
                <span>{{ minutes }} min</span>
              </p>
            </div>
          </article>
        {% endif %}
      {% endif %}
      {% if post.url == "/qwen-arch/" %}{% continue %}{% endif %}
      {% if post.url == "/pusht/" %}{% continue %}{% endif %}
      {% if post.url == "/cognateful/" %}{% continue %}{% endif %}
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
