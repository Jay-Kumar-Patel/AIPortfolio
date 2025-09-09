"use client";

import { useState, useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { ExternalLink } from "lucide-react";
import { SiLinkedin, SiGithub, SiGmail } from "react-icons/si";
import Image from "next/image";
import { MessageCircle, X, Send } from "lucide-react";

export default function Portfolio() {
  const bp = process.env.NEXT_PUBLIC_BASE_PATH || "";
  const withBase = (p: string) => `${bp}${p}`;

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [activeSection, setActiveSection] = useState("about");
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi 👋 I'm Jay!" },
    { role: "assistant", content: "Welcome to my AI assistant! Feel free to ask me anything about Jay's experience, projects, or skills. I will try to answer to the best of my knowledge." },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isChatOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isChatOpen]);

  useEffect(() => {
    const handleScroll = () => {
      const sections = ["about", "education", "experience", "projects"];
      const scrollPosition = window.scrollY + 100;
      for (let i = sections.length - 1; i >= 0; i--) {
        const section = document.getElementById(sections[i]);
        if (section) {
          const rect = section.getBoundingClientRect();
          const sectionTop = window.scrollY + rect.top;
          if (scrollPosition >= sectionTop) {
            setActiveSection(sections[i]);
            break;
          }
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const el = document.getElementById(sectionId);
    el?.scrollIntoView({ behavior: "smooth" });
  };

  const technologyLogos = [
    { name: "AWS", icon: "/aws.svg" },
    { name: "TypeScript", icon: "/typescript.svg" },
    { name: "Node", icon: "/node.svg" },
    { name: "Java", icon: "/java.svg" },
    { name: "Go", icon: "/go.svg" },
    { name: "Azure", icon: "/azure.svg" },
    { name: "Git", icon: "/git.svg" },
    { name: "React", icon: "/react.svg" },
    { name: "Terraform", icon: "/terraform.svg" },
    { name: "Docker", icon: "/docker.svg" },
    { name: "Kubernetes", icon: "/kubernetes.svg" },
    { name: "GCP", icon: "/gcp.svg" },
    { name: "Android Development", icon: "/android.svg" },
    { name: "Spring", icon: "/spring.svg" },
    { name: "Python", icon: "/python.svg" },
  ];

  const educationData = [
    {
      date: "2024 - 2025",
      course: "Master of Science",
      university: "Dalhousie University",
      degree: "Computer Science",
      logoUrl: "/dal_2.svg",
      link: "https://www.dal.ca/",
      subjects: [
        "Adv. Software Development",
        "Data Mgmt, Warhsng Analytics",
        "Adv. Cloud Computing",
        "Serverless Data Processing",
        "Adv. Cloud Architecting",
        "Adv. Topics Computer Networks",
        "Adv. Topics Network Security",
      ],
    },
    {
      date: "2019 - 2023",
      course: "Bachelor of Engineering",
      university: "Gujarat Technological University",
      degree: "Information and Communication Technology",
      logoUrl: "/gtu_2.svg",
      link: "https://www.gtu.ac.in/",
      subjects: [
        "Data Structures",
        "Analysis and Design of Algorithms",
        "Database Management Systems",
        "Operating System",
        "Computer Organization & Architecture",
        "Software Engineering",
        "Artificial Intelligence",
        "Machine Learning",
        "Cloud Computing",
        "Big Data Analytics",
      ],
    },
  ];

  const experienceData = [
    {
      date: "MAY 2025 - AUG 2025",
      name: "Software Engineer Co-Op - ApplyBoard",
      logoUrl: "/applyboard.svg",
      link: "https://www.applyboard.com/",
      details:
        "Built and deployed a serverless automation service to replace a legacy integration and reduce maintenance costs. Rolled out custom VPC domain solutions for internal APIs, migrating multiple services from public endpoints to secure private APIs. Enhanced security and observability with AWS X-Ray, reducing issue resolution time.",
      technologies: [
        "AWS",
        "TypeScript",
        "Node",
        "React",
        "AWS CDK",
        "GitHub Actions",
        "Infrastructure-as-Code",
      ],
    },
    {
      date: "OCT 2023 - MAY 2024",
      name: "Full Stack Developer - TronTechh LLC",
      logoUrl: "/primiv_2.svg",
      link: "https://www.primivtech.com/",
      details:
        "Developed and maintained a POS system supporting sales, inventory, and staff management with real-time access and updates. Integrated slot machine data from Arduino devices via the MQTT protocol to provide scheduled and historical reporting, improving accuracy and preventing manual errors. Built CI/CD pipelines with Jenkins to roll out the POS system across multiple stores, ensuring secure, high-performance operations and faster updates.",
      technologies: [
        "AWS",
        "Go",
        "React",
        "Java",
        "Spring Boot",
        "MySQL",
        "Jenkins",
        "Docker",
        "Kubernetes",
        "Microservices",
        "MQTT Protocol",
        "Arduino",
      ],
    },
    {
      date: "FEB 2023 - AUG 2023",
      name: "Android Developer Intern - Galanto Innovations",
      logoUrl: "/galanto.svg",
      link: "https://www.galantoindia.com/",
      details:
        "Developed and maintained a healthcare Android app integrating real-time smart glove sensor data for 24/7 patient monitoring. Built backend services for real-time synchronization and implemented responsive UI with interactive data visualizations to support clinical insights. Integrated Firebase, Bluetooth, and OTA updates to ensure secure communication, optimized performance, and reliable data processing.",
      technologies: [
        "Java",
        "XML",
        "Mobile Application Development",
        "Firebase",
        "MPAndroidChart",
        "Jetpack Compose",
        "Lottie",
        "Android Studio",
      ],
    },
  ];

  const projectsData = [
    {
      name: "RoomM8",
      logoUrl: "/roomM8.png",
      link: "https://github.com/Jay-Kumar-Patel/RoomM8",
      details:
        "RoomM8 is a full-stack roommate collaboration platform that ends the “who paid what / who’s doing what” chaos by centralizing expenses, tasks, announcements, and grocery lists into private, room-scoped workspaces accessed via unique invite codes. I containerized the app, automated delivery with CI/CD, and provisioned the entire environment with infrastructure as code aligned to the AWS Well-Architected Framework.",
      technologies: [
        "AWS",
        "Next.js",
        "Spring Boot",
        "Java",
        "MySQL",
        "Docker",
        "TypeScript",
        "GitLab CI/CD",
        "Test-Driven Development",
        "SOLID Principles",
        "Software Design Patterns",
        "Infrastructure-as-Code",
      ],
    },
    {
      name: "QuickData Processor",
      logoUrl: "/QuickData_Processor.png",
      link: "https://github.com/Jay-Kumar-Patel/QuickData-Processor",
      details:
        "QuickData Processor is a serverless multi-cloud (AWS and GCP) platform that turns raw JSON into insights by automating CSV conversion, NLP entity extraction, and word cloud analytics with a real-time admin dashboard. It uses event-driven services with secure multi-step authentication, asynchronous notifications, and a Dialogflow assistant, and ships via containerized builds with Terraform IaC for repeatable cross-cloud deployments.",
      technologies: [
        "AWS",
        "GCP",
        "Serverless Architecture",
        "Cross-Cloud Integration",
        "Looke Studio",
        "Firebase",
      ],
    },
    {
      name: "TinyDB",
      logoUrl: "/TinyDB.png",
      link: "https://github.com/Jay-Kumar-Patel/TinyDB",
      details:
        "TinyDB is a lightweight, MySQL-style relational database built in Java with a CLI for DDL/DML and ACID transactions (commit/rollback). It enforces core constraints (PK/FK/UNIQUE/NOT NULL), adds 2FA security, and auto-generates ERDs with SQL export for smooth migration.",
      technologies: ["Java"],
    },
    {
      name: "Slackify",
      logoUrl: "/slackify.png",
      link: "https://github.com/Jay-Kumar-Patel/Slackify",
      details:
        "Slackify is a cloud-native platform that seamlessly connects GitHub events with Slack to enable real-time collaboration and automated issue tracking. The infrastructure is deployed using Infrastructure as Code (IaC) with CloudFormation. This helps teams stay instantly informed, reduce response times, and improve overall productivity.",
      technologies: [
        "AWS",
        "Slack API",
        "GitHub WebHook",
        "Docker",
        "Infrastructure-as-Code",
        "Java",
        "Spring Boot",
      ],
    },
  ];

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = { role: "user", content: inputMessage };
    const currentInput = inputMessage;

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: currentInput }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
        },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I encountered an error while processing your question. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-10 sm:pt-12 lg:pt-16 bg-white">
      {/* Chat Window */}
      {isChatOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-[90vw] sm:w-96 md:w-[450px] h-[60vh] sm:h-[500px] bg-white border border-gray-200 rounded-lg shadow-xl flex flex-col">
          {/* Chat Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-[#3f3d56] text-white rounded-t-lg">
            <h3 className="text-lg font-semibold">Chat now</h3>
            <button
              onClick={() => setIsChatOpen(false)}
              className="text-white hover:text-gray-300 transition-colors"
              aria-label="Close chat"
            >
              <X size={24} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] p-3 rounded-lg ${
                    message.role === "user"
                      ? "bg-[#3f3d56] text-white"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  <p className="text-sm sm:text-base">{message.content}</p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 p-3 rounded-lg">
                  <p className="text-sm sm:text-base">Thinking...</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex gap-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                placeholder="Ask about Jay's portfolio..."
                className="flex-1 p-3 border border-gray-300 rounded-lg text-sm sm:text-base focus:outline-none focus:ring-2 focus:ring-[#3f3d56]"
                disabled={isLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading || !inputMessage.trim()}
                className="px-4 py-3 bg-[#3f3d56] text-white rounded-lg hover:bg-[#2d2b3d] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                aria-label="Send message"
              >
                <Send size={20} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Chat Button */}
      <button
        onClick={() => setIsChatOpen(true)}
        className={`fixed bottom-6 right-6 z-50 bg-[#3f3d56] text-white p-4 sm:p-5 rounded-full shadow-lg hover:bg-[#2d2b3d] transition-all duration-200 hover:scale-110 ${
          isChatOpen ? "hidden" : "block"
        }`}
        aria-label="Open chat"
      >
        <MessageCircle size={24} />
      </button>
      <div className="mx-auto w-full max-w-screen-2xl xl:max-w-[1600px] px-3 sm:px-6 lg:px-10 flex flex-col lg:flex-row lg:gap-24 xl:gap-28">
        {/* Left Sidebar - Desktop Only */}
        <div className="hidden lg:block w-80 h-screen p-8 flex flex-col sticky top-0">
          <div className="mb-8 flex flex-col items-start">
            <div className="w-32 h-32 rounded-full mb-6 overflow-hidden">
              <Image
                src={withBase("/profile.jpeg")}
                alt="Profile"
                width={128}
                height={128}
                className="object-cover rounded-full"
              />
            </div>

            <h1 className="mb-2 leading-tight text-[40px] font-[800]">
              Jay Patel
            </h1>

            <p className="mb-4 text-[16px] leading-snug font-[600]">
              Software Engineer
            </p>

            <p className="mb-4 text-[16px] leading-relaxed">
              I build secure, reliable, and scalable software platforms by
              combining cloud architecture, automation, and system design best
              practices.
            </p>
          </div>

          <nav className="mb-8">
            {[
              { id: "about", label: "ABOUT" },
              { id: "education", label: "EDUCATION" },
              { id: "experience", label: "EXPERIENCE" },
              { id: "projects", label: "PROJECTS" },
            ].map(({ id, label }) => (
              <button
                key={id}
                onClick={() => scrollToSection(id)}
                className={`block w-full text-left py-2 text-sm font-medium transition-colors ${
                  activeSection === id ? "border-b-2" : "hover:opacity-80"
                }`}
                style={{
                  borderColor: activeSection === id ? "#3f3d56" : "transparent",
                  color: activeSection === id ? "#3f3d56" : "#8b8a9a",
                }}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="flex gap-4 justify-start">
            <a
              href="https://www.linkedin.com/in/jaykumarpatel411/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#3f3d56] hover:opacity-80 transition-colors"
              aria-label="LinkedIn"
            >
              <SiLinkedin size={24} />
            </a>
            <a
              href="https://github.com/Jay-Kumar-Patel"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#3f3d56] hover:opacity-80 transition-colors"
              aria-label="GitHub"
            >
              <SiGithub size={24} />
            </a>
            <a
              href="mailto:jaykumarpatel111296@gmail.com"
              className="text-[#3f3d56] hover:opacity-80 transition-colors"
              aria-label="Email"
            >
              <SiGmail size={24} />
            </a>
          </div>
        </div>

        {/* Right Column (includes mobile header) */}
        <div className="flex-1 p-4 lg:p-8 pt-8 lg:pt-8 lg:ml-[5.5rem] xl:ml-[7.5rem] 2xl:ml-[9rem]">
          {/* Mobile Header */}
          <div className="lg:hidden text-center mb-12">
            <div className="w-32 h-32 rounded-full mx-auto mb-4 overflow-hidden">
              <Image
                src={withBase("/profile.jpeg")}
                alt="Profile"
                width={128}
                height={128}
                className="object-cover rounded-full"
              />
            </div>
            <h1 className="text-[32px] leading-tight font-[800] mb-3">
              Jay Patel
            </h1>
            <p className="text-[16px] leading-snug font-[600] mb-3">
              Software Engineer
            </p>
            <p className="text-[15px] leading-relaxed mb-6">
              I build secure, reliable, and scalable software platforms by
              combining cloud architecture, automation, and system design best
              practices.
            </p>
            <div className="flex w-full justify-center gap-4">
              <a
                href="https://www.linkedin.com/in/jaykumarpatel411/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#3f3d56] hover:opacity-80 transition-colors"
                aria-label="LinkedIn"
              >
                <SiLinkedin size={24} />
              </a>
              <a
                href="https://github.com/Jay-Kumar-Patel"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#3f3d56] hover:opacity-80 transition-colors"
                aria-label="GitHub"
              >
                <SiGithub size={24} />
              </a>
              <a
                href="mailto:jaykumarpatel111296@gmail.com"
                className="text-[#3f3d56] hover:opacity-80 transition-colors"
                aria-label="Email"
              >
                <SiGmail size={24} />
              </a>
            </div>
          </div>

          {/* ABOUT */}
          <section id="about" className="mb-12 lg:mb-16">
            <h2 className="mb-6 tracking-wide text-xl lg:text-2xl font-[800] antialiased">
              ABOUT ME
            </h2>

            <div className="space-y-6 text-sm sm:text-[15px] lg:text-base leading-relaxed">
              <p>
                My journey into software engineering began not in a classroom,
                but with a simple real-world problem: finding a way to notify
                vehicle owners when their car was parked in the wrong spot and
                at risk of being towed. I built an Android app that could
                quickly capture and recognize license plates with a phone
                camera, teaching myself how to combine image-to-text tools with
                mobile development. That experience showed me how exciting it is
                to take an idea from scratch and make it work in the real world.
              </p>

              <p>
                Now I’m finishing my Master’s in Applied Computer Science at{" "}
                <span className="font-semibold">Dalhousie University</span>.
                Along the way, I’ve worked on projects ranging from serverless
                data platforms and{" "}
                <span className="font-semibold">Full-Stack</span> web
                applications to{" "}
                <span className="font-semibold">Cloud Architecture</span> for
                scalable systems. I enjoy working on the backend and in the
                cloud, but what really excites me is turning an idea into
                something real that people can actually use.
              </p>

              <p>
                Outside of classes and co-ops, I spend time exploring{" "}
                <span className="font-semibold">Automation</span>, cloud
                workflows, and trading ideas with friends on how to build better
                systems. For me, software isn’t just code, it’s about solving
                problems in a way that’s simple, secure, and reliable.
              </p>
            </div>

            {/* Tech Logos - Mobile */}
            <div className="lg:hidden mt-12 mb-16">
              <div className="grid grid-cols-5 gap-4 justify-items-center">
                {technologyLogos.map((tech) => (
                  <button
                    key={tech.name}
                    type="button"
                    className="relative group h-10 w-10 focus:outline-none"
                    aria-label={tech.name}
                  >
                    <img
                      src={withBase(tech.icon)}
                      alt={`${tech.name} icon`}
                      className="h-full w-full object-contain transition-transform duration-150 active:scale-95 group-active:scale-110 group-focus:scale-110"
                      loading="lazy"
                      draggable={false}
                    />
                    <span
                      className="pointer-events-none absolute left-1/2 -translate-x-1/2 -top-2 -translate-y-full
                        rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5
                        text-[11px] font-medium text-emerald-700 whitespace-nowrap shadow-sm
                        opacity-0 transition-all duration-150
                        group-active:opacity-100 group-focus:opacity-100
                        group-active:-translate-y-[110%] group-focus:-translate-y-[110%]"
                    >
                      {tech.name}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Tech Logos - Desktop */}
            <div className="hidden lg:block mt-12 mb-16">
              <div className="flex flex-wrap gap-6 justify-center">
                {technologyLogos.map((tech) => (
                  <div key={tech.name} className="relative group h-10 w-10">
                    <img
                      src={withBase(tech.icon)}
                      alt={`${tech.name} icon`}
                      className="h-full w-full object-contain transition-transform duration-150 group-hover:scale-110 group-hover:drop-shadow"
                      loading="lazy"
                      draggable={false}
                    />
                    <span
                      className="pointer-events-none absolute left-1/2 -translate-x-1/2 -top-2 -translate-y-full
                        rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5
                        text-[11px] font-medium text-emerald-700 whitespace-nowrap shadow-sm
                        opacity-0 transition-all duration-150
                        group-hover:opacity-100 group-hover:-translate-y-[110%]"
                    >
                      {tech.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* CERTIFICATIONS */}
          <section id="certifications" className="mb-16">
            <h2 className="font-[800] mb-8 tracking-wide text-xl lg:text-2xl antialiased">
              CERTIFICATIONS
            </h2>

            <div className="space-y-8">
              <div className="block hover:scale-105 hover:shadow-lg hover:ring-2 hover:ring-black transition-all duration-200 p-4 rounded-lg">
                {/* Mobile */}
                <div className="lg:hidden">
                  <div className="flex items-start gap-4">
                    <img
                      src={withBase("/aws.svg")}
                      alt="AWS logo"
                      className="w-12 h-12 object-contain rounded-md"
                    />

                    <div className="flex-1">
                      <a
                        href="https://www.credly.com/badges/5d2af1b8-d81d-4919-84c5-d982ae224321/public_url" /* replace with your credential URL */
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group inline-flex items-center gap-2 text-base font-semibold leading-snug hover:text-blue-600 transition-colors"
                      >
                        <span>
                          AWS Certified Solutions Architect – Associate
                        </span>
                        <ExternalLink
                          size={16}
                          className="text-slate-400 group-hover:text-blue-600"
                        />
                      </a>

                      <p className="text-sm font-medium text-slate-700">
                        Amazon Web Services (AWS)
                      </p>
                      <p className="text-sm font-regular text-slate-600 mt-1">
                        Issued Aug 2025 · Expires Aug 2028
                      </p>
                    </div>
                  </div>
                </div>

                {/* Desktop */}
                <div className="hidden lg:block">
                  <div className="flex items-center gap-6">
                    <img
                      src={withBase("/aws.svg")}
                      alt="AWS logo"
                      className="w-14 h-14 object-contain rounded-md"
                    />

                    <div className="flex-1">
                      <a
                        href="https://www.credly.com/badges/5d2af1b8-d81d-4919-84c5-d982ae224321/public_url" /* replace with your credential URL */
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group inline-flex items-center gap-2 text-xl font-semibold hover:text-blue-600 transition-colors"
                      >
                        <span>
                          AWS Certified Solutions Architect – Associate
                        </span>
                        <ExternalLink
                          size={16}
                          className="text-slate-400 group-hover:text-blue-600"
                        />
                      </a>

                      <p className="text-base font-medium text-slate-700">
                        Amazon Web Services (AWS)
                      </p>
                      <p className="text-sm font-regular text-slate-600 mt-0.5">
                        Issued Aug 2025 · Expires Aug 2028
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* EDUCATION */}
          <section id="education" className="mb-16">
            <h2 className="font-[800] mb-8 tracking-wide text-xl lg:text-2xl">
              EDUCATION
            </h2>
            <div className="space-y-8">
              {educationData.map((item, idx) => (
                <div
                  key={idx}
                  className="block hover:scale-105 hover:shadow-lg hover:ring-2 hover:ring-black transition-all duration-200 p-4 rounded-lg"
                >
                  {/* Mobile */}
                  <div className="lg:hidden">
                    <div className="flex justify-center mb-4">
                      <img
                        src={withBase(item.logoUrl) || "/placeholder.svg"}
                        alt={`${item.university} logo`}
                        className="w-24 h-24 object-contain"
                      />
                    </div>

                    <div className="font-[700] mb-2 text-center text-lg">
                      {item.date}
                    </div>

                    {item.link ? (
                      <a
                        href={item.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex w-fit items-center justify-center gap-2 mb-4 mx-auto
               text-base font-semibold leading-snug tracking-tight
               text-[#2c2a3b] hover:text-blue-600 transition-colors"
                      >
                        <span>
                          {item.course} - {item.university}
                        </span>
                        <ExternalLink
                          size={16}
                          className="text-slate-400 group-hover:text-blue-600"
                        />
                      </a>
                    ) : (
                      <h3 className="text-base font-semibold leading-snug tracking-tight text-center">
                        {item.course} - {item.university}
                      </h3>
                    )}

                    <p className="text-alrge font-medium mb-6 text-center">
                      {item.degree}
                    </p>

                    <div className="flex flex-wrap gap-2 justify-start">
                      {item.subjects.map((subject) => (
                        <Badge
                          key={subject}
                          className="bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 text-xs font-medium px-2 py-0.5 rounded-md"
                        >
                          {subject}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Desktop */}
                  <div className="hidden lg:block">
                    <div className="lg:flex lg:gap-8">
                      <div className="lg:w-44 xl:w-48 lg:flex-shrink-0 lg:flex lg:flex-col">
                        <div className="text-l font-[700] whitespace-nowrap mb-2">
                          {item.date}
                        </div>
                        <div className="flex justify-center lg:justify-start">
                          <img
                            src={withBase(item.logoUrl) || "/placeholder.svg"}
                            alt={`${item.university} logo`}
                            className="w-24 h-24 object-contain"
                          />
                        </div>
                      </div>
                      <div className="flex-1 text-center lg:text-left">
                        {item.link ? (
                          <a
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group inline-flex items-center justify-center lg:justify-start gap-2 mb-2
                              text-xl font-semibold leading-snug tracking-tight
                              text-[#2c2a3b] hover:text-blue-600 transition-colors"
                          >
                            <span>
                              {item.course} - {item.university}
                            </span>
                            <ExternalLink
                              size={16}
                              className="text-slate-400 group-hover:text-blue-600"
                            />
                          </a>
                        ) : (
                          <h3 className="text-xl font-semibold leading-snug tracking-tight">
                            {item.course} - {item.university}
                          </h3>
                        )}
                        <p className="text-base font-medium mb-4">
                          {item.degree}
                        </p>
                        <div className="flex flex-wrap gap-2 justify-center lg:justify-start">
                          {item.subjects.map((subject) => (
                            <Badge
                              key={subject}
                              className="bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 text-sm"
                            >
                              {subject}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* EXPERIENCE */}
          <section id="experience" className="mb-16">
            <h2 className="font-[800] mb-8 tracking-wide text-xl lg:text-2xl">
              EXPERIENCE
            </h2>
            <div className="space-y-8">
              {experienceData.map((item, idx) => (
                <div
                  key={idx}
                  className="block hover:scale-105 hover:shadow-lg hover:ring-2 hover:ring-black transition-all duration-200 p-4 rounded-lg"
                >
                  {/* Mobile */}
                  <div className="lg:hidden">
                    <div className="flex justify-center mb-4">
                      <img
                        src={withBase(item.logoUrl) || "/placeholder.svg"}
                        alt={`${item.name} logo`}
                        className="w-24 h-24 object-contain"
                      />
                    </div>

                    {/* Centered date, same scale as Education mobile */}
                    <div className="font-[700] mb-2 text-center text-lg">
                      {item.date}
                    </div>

                    {/* Centered, clickable heading like Education mobile */}
                    {item.link ? (
                      <a
                        href={item.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex w-fit items-center justify-center gap-2 mb-4 mx-auto
                         text-base font-semibold leading-snug tracking-tight
                         hover:text-blue-600 transition-colors"
                      >
                        <span>{item.name}</span>
                        <ExternalLink
                          size={16}
                          className="text-slate-400 group-hover:text-blue-600"
                        />
                      </a>
                    ) : (
                      <h3 className="text-base font-semibold leading-snug tracking-tight text-center">
                        {item.name}
                      </h3>
                    )}

                    {/* Match smaller paragraph scale */}
                    <p className="text-sm leading-6 mb-4">{item.details}</p>

                    {/* Tags start from left, slightly smaller like Education mobile */}
                    <div className="flex flex-wrap gap-2 justify-start">
                      {item.technologies.map((tech) => (
                        <Badge
                          key={tech}
                          className="bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100
                           text-xs font-medium px-2 py-0.5 rounded-md"
                        >
                          {tech}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Desktop (unchanged layout; fonts aligned to Education desktop) */}
                  <div className="hidden lg:block">
                    <div className="lg:flex lg:gap-8">
                      <div className="lg:w-44 xl:w-48 lg:flex-shrink-0 lg:flex lg:flex-col">
                        <div className="text-l font-[700] whitespace-nowrap mb-2">
                          {item.date}
                        </div>
                        <div className="flex justify-center lg:justify-start">
                          <img
                            src={withBase(item.logoUrl) || "/placeholder.svg"}
                            alt={`${item.name} logo`}
                            className="w-24 h-24 object-contain"
                          />
                        </div>
                      </div>

                      <div className="flex-1 text-center lg:text-left">
                        {item.link ? (
                          <a
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group inline-flex items-center justify-center lg:justify-start gap-2 mb-3
                             text-xl font-semibold hover:text-blue-600 transition-colors"
                          >
                            <span>{item.name}</span>
                            <ExternalLink
                              size={16}
                              className="text-slate-400 group-hover:text-blue-600"
                            />
                          </a>
                        ) : (
                          <h3 className="text-xl font-semibold mb-3">
                            {item.name}
                          </h3>
                        )}

                        <p className="text-base leading-7 mb-4">
                          {item.details}
                        </p>

                        <div className="flex flex-wrap gap-2 justify-center lg:justify-start">
                          {item.technologies.map((tech) => (
                            <Badge
                              key={tech}
                              className="bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 text-sm"
                            >
                              {tech}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* PROJECTS */}
          <section id="projects" className="mb-16">
            <h2 className="font-[800] mb-8 tracking-wide text-xl lg:text-2xl">
              PROJECTS
            </h2>
            <div className="space-y-8">
              {projectsData.map((item, idx) => (
                <div
                  key={idx}
                  className="block hover:scale-105 hover:shadow-lg hover:ring-2 hover:ring-black transition-all duration-200 p-4 rounded-lg"
                >
                  {/* Mobile */}
                  <div className="lg:hidden">
                    <div className="flex justify-center mb-4">
                      <img
                        src={withBase(item.logoUrl) || "/placeholder.svg"}
                        alt={`${item.name} logo`}
                        className="w-24 h-24 object-contain"
                      />
                    </div>

                    {/* Centered, clickable heading like Education mobile */}
                    {item.link ? (
                      <a
                        href={item.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex w-fit items-center justify-center gap-2 mb-4 mx-auto text-lg lg:text-xl font-bold leading-snug tracking-tight hover:text-blue-600 transition-colors"
                      >
                        <span>{item.name}</span>
                        <ExternalLink
                          size={16}
                          className="text-slate-400 group-hover:text-blue-600"
                        />
                      </a>
                    ) : (
                      <h3 className="text-base font-semibold leading-snug tracking-tight text-center">
                        {item.name}
                      </h3>
                    )}

                    {/* Match smaller paragraph scale */}
                    <p className="text-sm leading-6 mb-4">{item.details}</p>

                    {/* Tags start from left, slightly smaller like Education mobile */}
                    <div className="flex flex-wrap gap-2 justify-start">
                      {item.technologies.map((tech) => (
                        <Badge
                          key={tech}
                          className="bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100
                           text-xs font-medium px-2 py-0.5 rounded-md"
                        >
                          {tech}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Desktop (unchanged layout; fonts aligned to Education desktop) */}
                  <div className="hidden lg:block">
                    <div className="lg:flex lg:gap-8">
                      <div className="lg:w-44 xl:w-48 lg:flex-shrink-0 lg:flex lg:flex-col">
                        <div className="flex justify-center lg:justify-start">
                          <img
                            src={withBase(item.logoUrl) || "/placeholder.svg"}
                            alt={`${item.name} logo`}
                            className="w-35 h-auto object-contain"
                          />
                        </div>
                      </div>

                      <div className="flex-1 text-center lg:text-left">
                        {item.link ? (
                          <a
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group inline-flex items-center justify-center lg:justify-start gap-2 mb-3
                             text-xl font-semibold hover:text-blue-600 transition-colors"
                          >
                            <span>{item.name}</span>
                            <ExternalLink
                              size={16}
                              className="text-slate-400 group-hover:text-blue-600"
                            />
                          </a>
                        ) : (
                          <h3 className="text-xl font-semibold mb-3">
                            {item.name}
                          </h3>
                        )}

                        <p className="text-base leading-7 mb-4">
                          {item.details}
                        </p>

                        <div className="flex flex-wrap gap-2 justify-center lg:justify-start">
                          {item.technologies.map((tech) => (
                            <Badge
                              key={tech}
                              className="bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 text-sm"
                            >
                              {tech}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
