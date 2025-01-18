from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from fuzzywuzzy import process
from gensim.models import KeyedVectors

app = Flask(__name__)
CORS(app) 

try:
    model = KeyedVectors.load_word2vec_format('C:/xampp/htdocs/word2vec-GoogleNews-vectors-master/GoogleNews-vectors-negative300.bin', binary=True)
except Exception as e:
    print(f"Error loading Word2Vec model: {e}")
    model = None  

jobs = [

    {"title": "Data Scientist", "description": "Python, Machine Learning, Data Analysis"},
    {"title": "Web Developer", "description": "HTML, CSS, JavaScript, React"},
    {"title": "AI Engineer", "description": "Deep Learning, TensorFlow, Python"},
    {"title": "Data Analyst", "description": "Excel, SQL, Data Visualization, Python"},
    {"title": "Cybersecurity Specialist", "description": "Network Security, Penetration Testing, Risk Assessment"},
    {"title": "Game Developer", "description": "Unity, Unreal Engine, Game Physics, C++"},
    {"title": "UX/UI Designer", "description": "Wireframing, Prototyping, Adobe XD, Figma, User Research"},
    
    {"title": "Product Manager", "description": "Agile Methodology, Product Roadmap, Stakeholder Management"},
    {"title": "Digital Marketing Specialist", "description": "SEO, Social Media Marketing, Google Ads, Content Creation"},
    {"title": "Business Analyst", "description": "Business Strategy, Data Modeling, Stakeholder Communication"},
    {"title": "Sales Manager", "description": "Sales Strategy, CRM Tools, Customer Relationship, Lead Generation"},
    {"title": "Human Resources Manager", "description": "Recruitment, Employee Relations, Talent Management, Onboarding"},
    
    {"title": "Graphic Designer", "description": "Adobe Photoshop, Illustrator, Creative Design, Typography"},
    {"title": "Content Writer", "description": "Copywriting, Blogging, Editing, SEO Content"},
    {"title": "Photographer", "description": "Photo Editing, Adobe Lightroom, Creative Vision, Camera Operation"},
    {"title": "Film Director", "description": "Scriptwriting, Cinematography, Directing, Film Editing"},
    {"title": "Social Media Manager", "description": "Content Planning, Analytics, Social Media Strategy, Branding"},

    {"title": "Registered Nurse", "description": "Patient Care, Nursing Procedures, Medication Administration"},
    {"title": "Pharmacist", "description": "Prescription Management, Drug Safety, Patient Counseling"},
    {"title": "Physical Therapist", "description": "Rehabilitation, Physical Therapy, Patient Care, Exercise Science"},
    {"title": "Dietitian", "description": "Nutritional Counseling, Meal Planning, Health Education"},
    {"title": "Medical Assistant", "description": "Patient Scheduling, Basic Clinical Procedures, Medical Records"},

    {"title": "School Teacher", "description": "Curriculum Planning, Classroom Management, Teaching"},
    {"title": "Academic Counselor", "description": "Student Guidance, Career Counseling, Academic Planning"},
    {"title": "University Lecturer", "description": "Subject Expertise, Research, Teaching University Students"},
    {"title": "Corporate Trainer", "description": "Employee Training, Workshop Facilitation, Skill Development"},

    {"title": "Hotel Manager", "description": "Guest Services, Staff Management, Revenue Optimization"},
    {"title": "Chef", "description": "Cooking, Menu Design, Kitchen Management, Food Presentation"},
    {"title": "Event Planner", "description": "Event Coordination, Budgeting, Vendor Management"},
    {"title": "Bartender", "description": "Mixology, Customer Service, Beverage Preparation"},

    {"title": "Musician", "description": "Instrumental Performance, Composition, Music Theory"},
    {"title": "Actor", "description": "Acting, Script Analysis, Stage Presence, Improvisation"},
    {"title": "Visual Artist", "description": "Painting, Drawing, Art Exhibitions, Creative Expression"},
    {"title": "Dancer", "description": "Choreography, Dance Performance, Physical Fitness"},

    {"title": "Electrician", "description": "Wiring, Electrical Systems, Troubleshooting, Safety Procedures"},
    {"title": "Carpenter", "description": "Woodworking, Construction, Blueprint Reading"},
    {"title": "Plumber", "description": "Pipe Installation, Repair, Plumbing Systems"},
    {"title": "Mechanic", "description": "Vehicle Maintenance, Engine Repair, Diagnostics"},

    {"title": "Accountant", "description": "Financial Reporting, Tax Preparation, Bookkeeping"},
    {"title": "Financial Analyst", "description": "Investment Analysis, Budget Planning, Financial Modeling"},
    {"title": "Lawyer", "description": "Legal Research, Case Management, Client Advocacy"},
    {"title": "Paralegal", "description": "Legal Documentation, Research, Administrative Support"},

    {"title": "Farmer", "description": "Crop Management, Livestock Care, Sustainable Farming Practices"},
    {"title": "Environmental Scientist", "description": "Environmental Impact Assessment, Data Collection, Research"},
    {"title": "Landscape Architect", "description": "Landscaping Design, Environmental Planning, Aesthetic Expertise"},
    {"title": "Fisheries Manager", "description": "Marine Ecosystems, Sustainable Fishing Practices, Conservation"},

    {"title": "Logistics Manager", "description": "Supply Chain Management, Warehousing, Distribution, ERP Tools"},
    {"title": "Warehouse Supervisor", "description": "Inventory Management, Staff Supervision, Logistics Coordination"},
    {"title": "Procurement Officer", "description": "Supplier Management, Contract Negotiation, Purchase Orders"},

    {"title": "Fitness Trainer", "description": "Exercise Programs, Personal Training, Nutrition Advice"},
    {"title": "Sports Coach", "description": "Team Leadership, Skill Development, Game Strategies"},
    {"title": "Yoga Instructor", "description": "Yoga Techniques, Meditation, Flexibility Training"},
]

skill_mapping = {

    "typing fast": "data entry",
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "javascript": "JavaScript",
    "html": "HTML",
    "css": "CSS",
    "machine learning": "Machine Learning",
    "data analysis": "Data Analysis",
    "deep learning": "Deep Learning",
    "sql": "SQL",
    "excel": "Excel",
    "data visualization": "Data Visualization",
    "web development": "Web Development",
    "software development": "Software Development",
    "cloud computing": "Cloud Computing",
    "cybersecurity": "Cybersecurity",
    "networking": "Networking",
    "mobile app development": "Mobile Development",
    "api development": "API Development",
    "devops": "DevOps",
    
    "analyzing documents": "data analysis",
    "statistical analysis": "Statistical Analysis",
    "market research": "Market Research",
    "business analysis": "Business Analysis",
    "financial analysis": "Financial Analysis",
    
    "graphic design": "Graphic Design",
    "content writing": "Content Writing",
    "copywriting": "Copywriting",
    "video editing": "Video Editing",
    "photography": "Photography",
    "social media management": "Social Media Management",
    "user experience design": "UX Design",
    "user interface design": "UI Design",
    
    "public speaking": "Public Speaking",
    "negotiation": "Negotiation",
    "presentation skills": "Presentation Skills",
    "customer service": "Customer Service",
    "team collaboration": "Team Collaboration",
    
    "project management": "Project Management",
    "time management": "Time Management",
    "strategic planning": "Strategic Planning",
    "team leadership": "Team Leadership",
    "stakeholder management": "Stakeholder Management",
    
    "fast runner": "athletics",
    "basketball": "sports",
    "writing code": "software development",
    "problem solving": "Problem Solving",
    "critical thinking": "Critical Thinking",
    "adaptability": "Adaptability",
    "creativity": "Creativity",
    "attention to detail": "Attention to Detail",
    
    "political analyst": "Political Science",
    "healthcare management": "Healthcare Management",
    "sales strategy": "Sales Strategy",
    "digital marketing": "Digital Marketing",
    "event planning": "Event Planning",
    "human resources": "Human Resources",
    "supply chain management": "Supply Chain Management",
    "real estate": "Real Estate",
    
    "yoga instructor": "Yoga Instruction",
    "fitness training": "Fitness Training",
    "language skills": "Language Proficiency",
    "research skills": "Research Skills",
    "data entry": "Data Entry",
    "customer relationship management": "CRM",
    "social media marketing": "Social Media Marketing",
    "search engine optimization": "SEO",
}

def normalize_skills(user_skills):
    normalized_skills = []
    for skill in user_skills.split(','):
        skill = skill.strip().lower()

        if skill in skill_mapping:
            normalized_skills.append(skill_mapping[skill])
        else:
            closest_match = process.extractOne(skill, skill_mapping.keys())
            if closest_match[1] > 80:
                normalized_skills.append(skill_mapping[closest_match[0]])
            else:

                if model is not None:
                    try:
                        similar_words = model.most_similar(skill, topn=3)
                        for word, _ in similar_words:
                            if word in skill_mapping:
                                normalized_skills.append(skill_mapping[word])
                                break
                    except KeyError:

                        normalized_skills.append(skill)
                else:
                    normalized_skills.append(skill) 
    
    return normalized_skills

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_skills = data.get('skills', '')

        print(f"User skills received: {user_skills}")

        if not user_skills.strip():
            print("No skills provided by the user.")
            return jsonify({"jobs": []})

        normalized_skills = normalize_skills(user_skills)
        print(f"Normalized skills: {normalized_skills}")

        job_df = pd.DataFrame(jobs)

        job_df.loc[len(job_df)] = ["User Input", ', '.join(normalized_skills)]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(job_df['description'])

        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        similar_indices = similarity.argsort()[0][-3:][::-1]
        recommended_jobs = job_df.iloc[similar_indices]['title'].tolist()

        print(f"Recommended jobs: {recommended_jobs}")

        return jsonify({"jobs": recommended_jobs})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "An error occurred during recommendation"}), 500

if __name__ == '__main__':
    app.run(debug=True)
