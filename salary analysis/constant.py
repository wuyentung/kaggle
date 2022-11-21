#%%
JOB_CAT = 'Job Category'

JOB_TITLE = 'Job Title'
SALARY_ESTIMATE = 'Salary Estimate'
JOB_DES = 'Job Description'
RATING = 'Rating'
CO_NAME = 'Company Name'
LOCATION = 'Location'
HEADQUARTERS = 'Headquarters'
SIZE = 'Size'
FOUNDED = 'Founded'
OWNERSHIP_TYPE = 'Type of ownership'
INDUSTRY = 'Industry'
SECTOR = 'Sector'
REVENUE = 'Revenue'
COMPETITORS = 'Competitors'
EASY_APPLY = 'Easy Apply'

SALARY_LOWER = 'Salary Lower'
SALARY_UPPER = 'Salary Upper'

SENIOR = 'Senior Level'
MID = 'Mid Level'
JUNIOR = 'Junior Level'

stopward_list = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "", 
            ]
def _form_skills() -> dict:
    def nothing():
        pass
    skill_types= {}
    ## assume all the skill is representative, which fulfills MECE principle
    skill_types['Statistics'] = ['matlab',
    'statistical',
    'models',
    'modeling',
    'statistics',
    'analytics',
    'forecasting',
    'predictive',
    'r',
    'control chart',
    'Julia']

    skill_types['Machine Learning'] = ['data robot',
    'tensorflow',
    'knime',
    'rapidminer',
    'mahout',
    'logical glue',
    'nltk',
    'networkx',
    'scikit',
    'torch',
    'keras',
    'caffe',
    'weka',
    'orange',
    'qubole',
    'ai',
    'nlp',
    'ml',
    'dl',
    'data min',
    'Machine Learning', 
    'neural network',
    'deep learning', 
    'spark',
    'mlops',
    ]

    skill_types['Data Visualization'] = ['tableau',
    'powerpoint',
    'Qlik',
    'looker',
    'powerbi',
    'matplotlib',
    'tibco',
    'bokeh',
    'd3',
    'octave',
    'shiny',
    'microstrategy']

    skill_types['Data Engineering'] = ['etl',
    'mining',
    'warehouse',
    'warehousing',
    'cloud',
    'sap',
    'salesforce',
    'openrefine',
    'redis',
    'sybase',
    'cassandra',
    'msaccess',
    'database',
    'aws',
    'ibm cloud',
    'azure',
    'redshift',
    's3',
    'ec2',
    'rds',
    'bigquery',
    'google cloud platform',
    'hadoop',
    'hive',
    'kafka',
    'hbase',
    'mesos',
    'pig',
    'storm',
    'scala',
    'hdfs',
    'mapreduce',
    'kinesis',
    'flink']

    skill_types['Software Engineering'] = ['java',
    'javascript',
    'c#',
    'c',
    'c++',
    'docker',
    'ansible',
    'jenkins',
    'nodejs',
    'angularjs',
    'css',
    'html',
    'terraform',
    'kubernetes',
    'lex',
    'perl',
    'cplusplus', 
    ]

    skill_types['SQL'] = ['sql',
    'oracle',
    'mysql',
    'oracle nosql',
    'nosql',
    'postgresql',
    'plsql',
    'mongodb']

    skill_types['Trait Skills'] = ['to Learning',
    'Time Management',
    'Attention to Detail',
    'Problem Solving',
    'critical thinking']

    skill_types['Social Skills']= ['teamwork',
    'team',
    'communication',
    'written',
    'verbal',
    'writing',
    'leadership',
    'interpersonal',
    'personal motivation',
    'storytelling']

    skill_types['Business'] = ['excel',
    'bi',
    'reporting',
    'reports',
    'dashboard',
    'business intelligence',
    'business']

    for k,v in skill_types.items():
        skill_types[k] = [skill.lower() for skill in skill_types.get(k)]
    return skill_types
skill_types = _form_skills()