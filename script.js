document.addEventListener("DOMContentLoaded", function() {
    const blogs = [
        {
            title: "Improving Rural Healthcare with AI",
            image: "images/ai-healthcare.jpg",
            description: "Discover how AI is transforming healthcare accessibility in rural areas."
        },
        {
            title: "Telemedicine: A Game Changer",
            image: "images/telemedicine.jpg",
            description: "How virtual consultations are making healthcare accessible in remote villages."
        },
        {
            title: "Low-Cost Medical Innovations",
            image: "images/low-cost-health.jpg",
            description: "Exploring cost-effective solutions for better healthcare facilities."
        }
    ];

    const blogContainer = document.getElementById("blog-list");

    blogs.forEach(blog => {
        let blogDiv = document.createElement("div");
        blogDiv.classList.add("blog");

        blogDiv.innerHTML = `
            <img src="${blog.image}" alt="${blog.title}">
            <h3>${blog.title}</h3>
            <p>${blog.description}</p>
        `;

        blogContainer.appendChild(blogDiv);
    });
});
