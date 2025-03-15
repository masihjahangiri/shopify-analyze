from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd

@dataclass
class App:
    """Shopify app data model."""
    id: int
    url: str
    title: str
    developer: str
    developer_link: str
    icon: str
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    description: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    
    # Additional derived fields
    key_benefits: List[str] = field(default_factory=list)
    pricing_plans: List[Any] = field(default_factory=list)
    reviews_summary: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'App':
        """Create an App instance from a DataFrame row."""
        app = cls(
            id=row.get('id'),
            url=row.get('url', ''),
            title=row.get('title', ''),
            developer=row.get('developer', ''),
            developer_link=row.get('developer_link', ''),
            icon=row.get('icon', ''),
            rating=row.get('rating_value') if 'rating_value' in row else row.get('rating'),
            reviews_count=row.get('reviews_count'),
            description=row.get('description'),
            categories=row.get('categories', []) if isinstance(row.get('categories'), list) else []
        )
        
        # Add reviews summary if available
        review_fields = ['review_count', 'avg_rating', 'min_rating', 'max_rating', 
                         'std_rating', 'developer_reply_count', 'reply_rate']
        for field in review_fields:
            if field in row:
                app.reviews_summary[field] = row[field]
        
        return app

@dataclass
class Category:
    """App category data model."""
    id: int
    title: str
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'Category':
        """Create a Category instance from a DataFrame row."""
        return cls(
            id=row['id'],
            title=row['title']
        )

@dataclass
class KeyBenefit:
    """App key benefit data model."""
    app_id: int
    description: str
    title: Optional[str] = None
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'KeyBenefit':
        """Create a KeyBenefit instance from a DataFrame row."""
        return cls(
            app_id=row['app_id'],
            title=row.get('title'),
            description=row['description']
        )

@dataclass
class PricingPlan:
    """App pricing plan data model."""
    id: int
    app_id: int
    title: str
    plan_type: Optional[str] = None
    features: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'PricingPlan':
        """Create a PricingPlan instance from a DataFrame row."""
        return cls(
            id=row['id'],
            app_id=row['app_id'],
            title=row['title'],
            plan_type=row.get('plan_type')
        )

@dataclass
class Review:
    """App review data model."""
    app_id: int
    author: str
    rating: float
    posted_at: Any  # Datetime
    body: Optional[str] = None
    helpful_count: Optional[int] = None
    developer_reply: Optional[str] = None
    developer_reply_posted_at: Optional[Any] = None  # Datetime
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'Review':
        """Create a Review instance from a DataFrame row."""
        return cls(
            app_id=row['app_id'],
            author=row['author'],
            rating=row['rating'],
            posted_at=row['posted_at'],
            body=row.get('body'),
            helpful_count=row.get('helpful_count'),
            developer_reply=row.get('developer_reply'),
            developer_reply_posted_at=row.get('developer_reply_posted_at')
        )

class ShopifyDataset:
    """Container for all Shopify app data."""
    
    def __init__(self):
        self.apps: Dict[int, App] = {}
        self.categories: Dict[int, Category] = {}
        self.app_categories: Dict[int, List[int]] = {}
        self.key_benefits: Dict[int, List[KeyBenefit]] = {}
        self.pricing_plans: Dict[int, List[PricingPlan]] = {}
        self.review_summaries: Dict[int, Dict[str, Any]] = {}
    
    def load_from_dataframes(self, 
                           apps_df: pd.DataFrame,
                           categories_df: pd.DataFrame,
                           apps_categories_df: pd.DataFrame,
                           key_benefits_df: Optional[pd.DataFrame] = None,
                           pricing_plans_df: Optional[pd.DataFrame] = None,
                           pricing_plan_features_df: Optional[pd.DataFrame] = None,
                           review_summaries_df: Optional[pd.DataFrame] = None):
        """Load data from pandas DataFrames."""
        # Load categories
        for _, row in categories_df.iterrows():
            category = Category.from_dataframe_row(row)
            self.categories[category.id] = category
        
        # Load app-category relationships
        for _, row in apps_categories_df.iterrows():
            app_id = row['app_id']
            category_id = row['category_id']
            
            if app_id not in self.app_categories:
                self.app_categories[app_id] = []
            
            self.app_categories[app_id].append(category_id)
        
        # Load key benefits if available
        if key_benefits_df is not None:
            for _, row in key_benefits_df.iterrows():
                benefit = KeyBenefit.from_dataframe_row(row)
                
                if benefit.app_id not in self.key_benefits:
                    self.key_benefits[benefit.app_id] = []
                
                self.key_benefits[benefit.app_id].append(benefit)
        
        # Load pricing plans and features if available
        if pricing_plans_df is not None:
            for _, row in pricing_plans_df.iterrows():
                plan = PricingPlan.from_dataframe_row(row)
                
                if plan.app_id not in self.pricing_plans:
                    self.pricing_plans[plan.app_id] = []
                
                self.pricing_plans[plan.app_id].append(plan)
            
            # Add features to pricing plans if available
            if pricing_plan_features_df is not None:
                plan_features = {}
                for _, row in pricing_plan_features_df.iterrows():
                    plan_id = row['pricing_plan_id']
                    feature = row['feature']
                    
                    if plan_id not in plan_features:
                        plan_features[plan_id] = []
                    
                    plan_features[plan_id].append(feature)
                
                # Add features to plans
                for app_plans in self.pricing_plans.values():
                    for plan in app_plans:
                        if plan.id in plan_features:
                            plan.features = plan_features[plan.id]
        
        # Load review summaries if available
        if review_summaries_df is not None:
            for _, row in review_summaries_df.iterrows():
                app_id = row['app_id']
                summary = {col: row[col] for col in review_summaries_df.columns if col != 'app_id'}
                self.review_summaries[app_id] = summary
        
        # Load apps and combine with related data
        for _, row in apps_df.iterrows():
            app_id = row['id']
            app = App.from_dataframe_row(row)
            
            # Add categories
            if app_id in self.app_categories:
                app.categories = [self.categories[cat_id].title 
                                  for cat_id in self.app_categories[app_id]
                                  if cat_id in self.categories]
            
            # Add key benefits
            if app_id in self.key_benefits:
                app.key_benefits = [benefit.description for benefit in self.key_benefits[app_id]]
            
            # Add pricing plans
            if app_id in self.pricing_plans:
                app.pricing_plans = self.pricing_plans[app_id]
            
            # Add review summary
            if app_id in self.review_summaries:
                app.reviews_summary = self.review_summaries[app_id]
            
            self.apps[app_id] = app
    
    def get_app(self, app_id: int) -> Optional[App]:
        """Get app by ID."""
        return self.apps.get(app_id)
    
    def get_apps_by_category(self, category_title: str) -> List[App]:
        """Get all apps in a given category."""
        return [app for app in self.apps.values() 
                if category_title in app.categories]
    
    def get_apps_by_developer(self, developer_name: str) -> List[App]:
        """Get all apps by a given developer."""
        return [app for app in self.apps.values() 
                if app.developer == developer_name] 