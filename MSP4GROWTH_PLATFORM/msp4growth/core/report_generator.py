"""
MSP4GROWTH Report Generator
Generates comprehensive PDF reports for MSP4GROWTH analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import contextily as ctx
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas
import tempfile
from ..config import RESULTS_DIR, DATA_DIR
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from .display_name_mapper import DisplayNameMapper
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

class NumberedCanvas(canvas.Canvas):
    """Custom canvas for adding page numbers and footers."""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.footer_text = "MSP4GROWTH Analysis Report - Generated on {}".format(
            datetime.now().strftime("%B %d, %Y")
        )

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for (page_num, page_state) in enumerate(self._saved_page_states):
            self.__dict__.update(page_state)
            self.draw_page_number(page_num + 1, num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_num, total_pages):
        """Draw page number and footer on each page."""
        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.grey)
        self.drawString(30, 20, self.footer_text)
        
        # Page number
        page_text = f"Page {page_num} of {total_pages}"
        self.drawRightString(A4[0] - 30, 20, page_text)

class MSP4GROWTHReportGenerator:
    """
    Generates comprehensive PDF reports for MSP4GROWTH analysis results.
    """
    
    def __init__(self, analysis_instance):
        """
        Initialize the report generator.
        
        Args:
            analysis_instance: MSP4GROWTHAnalysis instance
        """
        self.analysis = analysis_instance
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.temp_dir = Path(tempfile.mkdtemp())
        
                
        # Initialize display name mapper
        self.display_mapper = DisplayNameMapper()
        
        # Add any custom mappings specific to your project
        custom_mappings = {
            # Add your project-specific mappings here
            'vi': 'Suitability Value',
            'dki_matrix': 'Distance Matrix',
            'centroid_to_id': 'Centroid Mapping',
            # Add more as needed
        }
        self.display_mapper.add_custom_mapping(custom_mappings)

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86AB')
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2E86AB'),
            borderWidth=1,
            borderColor=colors.HexColor('#2E86AB'),
            borderPadding=5
        )
        
        # Body style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=13,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        # Caption style
        self.caption_style = ParagraphStyle(
            'Caption',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=6
        )

    def generate_report(self, result_paths: Dict[str, Path]) -> Path:
        """
        Generate the complete PDF report.
        
        Args:
            result_paths: Dictionary of model names to result file paths
            
        Returns:
            Path to the generated PDF report
        """
        logger.info("Starting PDF report generation")
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"MSP4GROWTH_Report_{self.analysis.request_id}_{timestamp}.pdf"
        report_path = self.analysis.results_dir / report_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=50
        )
        
        # Build story (content)
        story = []
        
        # Add all pages
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        story.extend(self._create_executive_summary())
        story.append(PageBreak())
        
        story.extend(self._create_data_parameters_page())
        story.append(PageBreak())
        
        # Add model results pages
        for model_name, result_path in result_paths.items():
            if result_path and result_path.exists():
                # Map visualization page
                story.extend(self._create_model_map_page(model_name, result_path))
                story.append(PageBreak())
                
                # Results table page
                story.extend(self._create_model_table_page(model_name, result_path))
                story.append(PageBreak())
        
        story.extend(self._create_disclaimer_page())
        story.append(PageBreak())
        
        story.extend(self._create_appendix())
        
        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        
        logger.info(f"PDF report generated successfully: {report_path}")
        return report_path

    def _create_title_page(self) -> List:
        """Create the title page."""
        story = []
        
        # Add logo if available
        logo_path = Path( DATA_DIR / "MSP4GROWTH-LOGO.png")  # Adjust path as needed
        if logo_path.exists():
            img = Image(str(logo_path), width=2*inch, height=1*inch)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 20))

        # Title
        title = Paragraph("MSP4GROWTH Analysis Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Use case and details
        use_case_text = f"<b>Use Case:</b> {self.display_mapper.get_display_name(self.analysis.use_case)}"
        story.append(Paragraph(use_case_text, self.body_style))
        
        request_id_text = f"<b>Request ID:</b> {self.analysis.request_id}"
        story.append(Paragraph(request_id_text, self.body_style))
        
        date_text = f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        story.append(Paragraph(date_text, self.body_style))
        
        story.append(Spacer(1, 40))
        
        # Abstract
        abstract = """
        This report presents the results of a comprehensive Maritime Spatial Planning analysis 
        conducted using the MSP4GROWTH framework. The analysis evaluates optimal spatial 
        allocation strategies based on multiple environmental, economic, and technical criteria 
        to support sustainable marine resource development.
        """
        story.append(Paragraph("<b>Abstract</b>", self.heading_style))
        story.append(Paragraph(abstract, self.body_style))
        
        return story

    def _create_executive_summary(self) -> List:
        """Create the executive summary page."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        # Analysis overview
        overview = f"""
        This Maritime Spatial Planning analysis was conducted for {self.analysis.use_case} 
        activities within the specified area of interest. The analysis employed multiple 
        optimization models to identify optimal spatial configurations that balance 
        environmental constraints with development objectives.
        """
        story.append(Paragraph(overview, self.body_style))
        
        # Key parameters
        story.append(Paragraph("<b>Key Analysis Parameters:</b>", self.body_style))
        
        params_data = [
            ["Parameter", "Value"],
            ["Use Case", self.display_mapper.get_display_name(self.analysis.use_case)],
            ["Calculation Type", self.display_mapper.get_display_name(self.analysis.calc_type)],
            ["Number of Optimal Sites (C)", str(self.analysis.C_number)],
            ["Size of Optimal Sitesints (N)", f"{self.analysis.N_size[0]} - {self.analysis.N_size[1]}"],
            ["Number of Datasets", str(len(self.analysis.thresholds))],
        ]
        
        params_table = Table(params_data, colWidths=[2*inch, 2*inch])
        params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(params_table)
        story.append(Spacer(1, 20))
        
        # Models summary
        models_text = """
        The analysis utilized advanced optimization algorithms to evaluate trade-offs between 
        competing objectives and identify Pareto-optimal solutions. Each model provides unique 
        insights into the spatial planning problem, allowing decision-makers to understand 
        the implications of different management strategies.
        """
        story.append(Paragraph("<b>Models Applied:</b>", self.body_style))
        story.append(Paragraph(models_text, self.body_style))
        
        return story

    def _create_data_parameters_page(self) -> List:
        """Create the data and parameters page."""
        story = []
        
        story.append(Paragraph("Data and Parameters", self.heading_style))
        
        # Area of Interest
        story.append(Paragraph("<b>Area of Interest</b>", self.body_style))
        
        
        # Calculate area of input polygon
        try:
            
            # Create polygon from coordinates
            aoi_polygon = Polygon(self.analysis.input_polygon)
            
            # Calculate area in square kilometers
            area_m2 = aoi_polygon.area
            area_km2 = area_m2 / 1_000_000
            
            aoi_text = f"""
            The analysis was conducted within a study area covering approximately {area_km2:.2f} km² 
            of marine space. This area encompasses the region designated for 
            {self.display_mapper.get_display_name(self.analysis.use_case)} development planning.
            The study area boundary was defined by {len(self.analysis.input_polygon)} coordinate points.
            """
            
        except Exception as e:
            logger.warning(f"Could not calculate polygon area: {str(e)}")
            # Fallback to coordinate count
            aoi_text = f"""
            The analysis was conducted within a polygon defined by {len(self.analysis.input_polygon)} 
            coordinate points. The area encompasses the marine space designated for 
            {self.display_mapper.get_display_name(self.analysis.use_case)} development planning.
            """

        story.append(Paragraph(aoi_text, self.body_style))
        
        # Datasets and thresholds
        story.append(Paragraph("<b>Datasets and Criteria</b>", self.body_style))
        
        datasets_data = [["Dataset", "Weight Coefficients", "Thresholds (Min-Max)", "Type"]]
        
        for dataset_name, config in self.analysis.thresholds.items():
            weight = config.get('weights', 0)
            thresholds = config.get('thresholds', [])
            threshold_range = f"{min(thresholds):.2f} - {max(thresholds):.2f}" if thresholds else "N/A"
            dataset_type = "Custom" if dataset_name in self.analysis.datasets else "Standard"
            
            display_name = self.display_mapper.get_display_name(dataset_name)

            datasets_data.append([
                display_name,
                f"{weight:.2f}",
                threshold_range,
                dataset_type
            ])
        
        datasets_table = Table(datasets_data, colWidths=[1.5*inch, 0.8*inch, 1.2*inch, 0.8*inch])
        datasets_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(datasets_table)
        story.append(Spacer(1, 20))
        
        # Optimization parameters
        story.append(Paragraph("<b>Optimization Parameters</b>", self.body_style))
        
        opt_params = f"""
        <b>Number of Optimal Sites (C):</b> {self.analysis.C_number}<br/>
        <b>Size of Optimal Sites (N):</b> Lower bound = {self.analysis.N_size[0]}, Upper bound = {self.analysis.N_size[1]}<br/>
        <b>Calculation Method:</b> {self.display_mapper.get_display_name(self.analysis.calc_type)}-based evaluation<br/>
        <b>Models Selected:</b> {', '.join([self.display_mapper.get_display_name(k) for k, v in self.analysis.model.items() if v > 0])}
        """
        story.append(Paragraph(opt_params, self.body_style))
        
        return story

    def _create_model_map_page(self, model_name: str, result_path: Path) -> List:
        """Create a map visualization page for a model."""
        story = []
        
        model_display_name = self.display_mapper.get_display_name(model_name)
        story.append(Paragraph(f"{model_display_name} - Spatial Results", self.heading_style))
        
        # Generate map
        map_path = self._generate_map(result_path, model_name)
        
        if map_path and map_path.exists():
            # Add map image
            img = Image(str(map_path), width=6*inch, height=4.5*inch)
            img.hAlign = 'CENTER'
            story.append(img)
            
            # Add caption
            caption_text = f"Figure: {model_display_name} model spatial optimization results showing selected areas overlaid on OpenStreetMap"
            story.append(Paragraph(caption_text, self.caption_style))
        # else:
        #     story.append(Paragraph("Map visualization could not be generated.", self.body_style))
        
        story.append(Spacer(1, 20))
        
        # Add model description
        model_descriptions = {
            'model_AUG': """
            The Augmented ε-constraint (AUG) method systematically explores the trade-off between 
            conflicting objectives by parametrically varying constraint bounds. This approach 
            generates a set of Pareto-optimal solutions, allowing decision-makers to understand 
            the relationship between different planning criteria.
            """,
            'model_WS': """
            The Weighted Sum (WS) method combines multiple objectives into a single composite 
            function using predetermined weights. This approach provides a single optimal solution 
            that represents the best compromise between competing objectives based on the specified 
            preference structure.
            """,
            'model_PF': """
            The Particle Filter (PF) method employs stochastic optimization techniques to explore 
            the solution space and identify high-quality spatial configurations. This approach 
            is particularly effective for handling uncertainty and complex constraint interactions.
            """
        }
        
        if model_name in model_descriptions:
            story.append(Paragraph("<b>Model Description:</b>", self.body_style))
            story.append(Paragraph(model_descriptions[model_name], self.body_style))
        
        return story

    def _create_model_table_page(self, model_name: str, result_path: Path) -> List:
        """Create a results table page for a model."""
        story = []
        
        model_display_name = self.display_mapper.get_display_name(model_name)
        story.append(Paragraph(f"{model_display_name} - Detailed Results", self.heading_style))

        try:
            # Load GeoJSON results
            gdf = gpd.read_file(result_path)
            
                        # Rename columns for display using the mapper
            display_columns_mapping = self.display_mapper.format_column_names(gdf.columns)
            gdf_display = gdf.rename(columns=display_columns_mapping)

            # Create summary statistics
            story.append(Paragraph("<b>Summary Statistics:</b>", self.body_style))
            
            summary_data = [
                ["Metric", "Value"],
                ["Total Selected Areas", str(len(gdf_display))],
                ["Total Area (km²)", f"{gdf.geometry.area.sum() / 1_000_000:.2f}"],
            ]
            
            # Add model-specific metrics if available
            if 'interest value' in gdf.columns:
                summary_data.append(["Average Suitability", f"{gdf['interest value'].mean():.2f}"])
                summary_data.append(["Max Suitability", f"{gdf['interest value'].max():.2f}"])
            
            summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Detailed results table
            story.append(Paragraph("<b>Detailed Results:</b>", self.body_style))
            
           # Select relevant columns for display (using original names to check)
            display_columns = []
            original_columns = []
            
            if 'fid' in gdf.columns:
                original_columns.append('fid')
                # reset fid to start from 1
                gdf['fid'] = gdf.index + 1
                display_columns.append(self.display_mapper.get_display_name('fid'))
            
            if 'interest value' in gdf.columns:
                original_columns.append('interest value')
                display_columns.append(self.display_mapper.get_display_name('interest value'))
            
            for threshold in self.analysis.thresholds.keys():
                if threshold in gdf.columns:
                    original_columns.append(threshold)
                    display_columns.append(self.display_mapper.get_display_name(threshold))
    
            # Create table data
            table_data = [display_columns]
            
            for idx, row in gdf.iterrows():
                row_data = []
                for col in original_columns:
                    if col in row:
                        if isinstance(row[col], (int, float)):
                            row_data.append(f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col]))
                        else:
                            row_data.append(str(row[col]))
                    else:
                        row_data.append("N/A")
                table_data.append(row_data)
                
                # # Limit to first 20 rows to prevent page overflow
                # if len(table_data) > 21:
                #     table_data.append(["...", "...", "..."])
                #     break
            
            results_table = Table(table_data, colWidths=[1*inch] * len(display_columns))
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(results_table)
            
        except Exception as e:
            logger.error(f"Error creating table for {model_name}: {str(e)}")
            story.append(Paragraph(f"Error loading results: {str(e)}", self.body_style))
        
        return story

    def _create_disclaimer_page(self) -> List:
        """Create the disclaimer page."""
        story = []
        
        story.append(Paragraph("Disclaimer", self.heading_style))
        
        disclaimer_text = """
        <b>Important Notice:</b><br/><br/>
        
        This report is generated automatically based on the MSP4GROWTH analysis framework and 
        should be interpreted by qualified marine spatial planning professionals. The results 
        presented herein are based on the datasets, parameters, and constraints specified during 
        the analysis setup.<br/><br/>
        
        <b>Limitations:</b><br/>
        • Results are dependent on the quality and completeness of input datasets<br/>
        • Optimization models provide mathematical solutions that may require additional 
        practical considerations<br/>
        • Environmental and regulatory constraints may change over time<br/>
        • Field verification is recommended before implementation<br/><br/>
        
        <b>Data Sources:</b><br/>
        All spatial datasets used in this analysis should be validated for currency and accuracy. 
        The MSP4GROWTH framework integrates multiple data sources, and users should verify 
        the suitability of these sources for their specific application.<br/><br/>
        
        <b>Technical Support:</b><br/>
        For questions regarding the methodology, data processing, or interpretation of results, 
        please contact the MSP4GROWTH development team.<br/><br/>
        
        <b>Citation:</b><br/>
        When referencing this report, please include the request ID ({}) and generation date.
        """.format(self.analysis.request_id)
        
        story.append(Paragraph(disclaimer_text, self.body_style))
        
        return story

    def _create_appendix(self) -> List:
        """Create the appendix."""
        story = []
        
        story.append(Paragraph("Appendix", self.heading_style))
        
        # Technical specifications
        story.append(Paragraph("<b>A. Technical Specifications</b>", self.body_style))
        
        tech_specs = """
        <b>Software Framework:</b> MSP4GROWTH v2.0<br/>
        <b>Coordinate System:</b> Web Mercator (EPSG:3857)<br/>
        <b>Data Format:</b> GeoJSON<br/>
        <b>Optimization Engine:</b> Gurobi Optimizer<br/>
        <b>Analysis Type:</b> Multi-criteria spatial optimization<br/>
        """
        story.append(Paragraph(tech_specs, self.body_style))
        
        # Model parameters
        story.append(Paragraph("<b>B. Model Parameters</b>", self.body_style))
        
        model_params = f"""
        The following parameters were used in the optimization models:<br/><br/>
        <b>General Parameters:</b><br/>
        • Request ID: {self.analysis.request_id}<br/>
        • Use Case: {self.display_mapper.get_display_name(self.analysis.use_case)}<br/>
        • Calculation Type: {self.display_mapper.get_display_name(self.analysis.calc_type)}<br/>
        • Number of Optimal Sites (C): {self.analysis.C_number}<br/>
        • Size of Optimal Sitesints (N): {self.analysis.N_size[0]} to {self.analysis.N_size[1]}<br/><br/>
        
        """
        
        story.append(Paragraph(model_params, self.body_style))
        
        # Methodology
        story.append(Paragraph("<b>C. Methodology Notes</b>", self.body_style))
        
        methodology = """
        <b>Multi-Criteria Decision Analysis:</b><br/>
        The MSP4GROWTH framework employs multi-criteria decision analysis (MCDA) techniques 
        to evaluate spatial alternatives based on multiple, often conflicting, criteria. 
        The approach integrates environmental, economic, and technical factors to identify 
        optimal spatial configurations for marine activities.<br/><br/>
        
        <b>Optimization Approach:</b><br/>
        Three complementary optimization methods are available:<br/>
        • Augmented ε-constraint method for Pareto frontier exploration<br/>
        • Weighted sum method for single-objective optimization<br/>
        • Particle filter method for stochastic optimization<br/><br/>
        
        <b>Spatial Analysis:</b><br/>
        All spatial operations are performed using geodetic calculations appropriate for 
        marine environments. Distance calculations and area computations account for the 
        Earth's curvature and are suitable for global applications.
        """
        
        story.append(Paragraph(methodology, self.body_style))
        
        return story
    
    def _generate_map(self, result_path: Path, model_name: str) -> Optional[Path]:
        """
        Generate a map visualization of the results.
        
        Args:
            result_path: Path to the GeoJSON results file
            model_name: Name of the model
            
        Returns:
            Path to the generated map image
        """
        try:
            # Load results
            gdf = gpd.read_file(result_path)

            if gdf.crs.to_epsg != 4326:
               gdf.to_crs(epsg=4326, inplace=True)
             
            if gdf.empty:
                logger.warning(f"No data found in {result_path}")
                return None
            
            # Calculate bounds and add zoom-out buffer
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            # Add significant buffer for zoomed-out view (50% on each side)
            x_range = bounds[2] - bounds[0]
            y_range = bounds[3] - bounds[1]
            buffer_x = x_range * 0.5  # 50% buffer on x-axis
            buffer_y = y_range * 0.5  # 50% buffer on y-axis
            
            # Apply buffer to bounds
            zoomed_bounds = [
                bounds[0] - buffer_x,  # minx with buffer
                bounds[1] - buffer_y,  # miny with buffer  
                bounds[2] + buffer_x,  # maxx with buffer
                bounds[3] + buffer_y   # maxy with buffer
            ]
            
            # Create high-quality figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=200)

            # Plot the results
            if 'interest value' in gdf.columns:
                # Color by suitability if available
                gdf.plot(ax=ax, column='interest value', 
                         cmap='viridis', 
                         legend=False, 
                         alpha=0.8, 
                         edgecolor='black', 
                         linewidth=1.5)
                
                # Create normalization and colormap
                norm = Normalize(vmin=0, vmax=4)
                colormap = cm.get_cmap('viridis')
                
                # Create colorbar manually
                mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
                mappable.set_array(gdf['interest value'])
                
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.02)
                cbar.set_label('Suitability Score', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
                
            else:
                # Simple plot if no suitability column
                gdf.plot(ax=ax, color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add FID labels centered on each polygon
            try:
                # Find FID column (case insensitive)
                fid_column = None
                for col in gdf.columns:
                    if col.lower() == 'fid':
                        fid_column = col
                        break
                
                if fid_column:
                    # Reset FID to start from 1 for display
                    display_fids = range(1, len(gdf) + 1)
                    
                    # Add text labels at polygon centroids
                    for idx, (_, row) in enumerate(gdf.iterrows()):
                        # Calculate centroid
                        centroid = row.geometry.centroid
                        
                        # Add text label
                        ax.text(centroid.x, centroid.y, 
                               str(display_fids[idx]),
                               fontsize=10,
                               fontweight='bold',
                               ha='center',
                               va='center',
                               color='white',
                               bbox=dict(boxstyle="circle,pad=0.3", 
                                       facecolor='black', 
                                       edgecolor='white',
                                       alpha=0.8,
                                       linewidth=1))
                else:
                    logger.warning("No FID column found for labeling polygons")
                    
            except Exception as e:
                logger.warning(f"Could not add FID labels to map: {str(e)}")

            # Set the zoomed-out bounds BEFORE adding basemap
            ax.set_xlim(zoomed_bounds[0], zoomed_bounds[2])
            ax.set_ylim(zoomed_bounds[1], zoomed_bounds[3])
            
            # Add professional basemap (OpenStreetMap) with zoomed-out view
            try:
                ctx.add_basemap(ax, 
                              crs=gdf.crs, 
                              source=ctx.providers.OpenStreetMap.Mapnik, 
                              alpha=0.85,
                              zoom='auto')  # Let contextily determine appropriate zoom
                logger.info("Successfully added OpenStreetMap basemap with zoomed-out view")
            except Exception as e:
                logger.warning(f"Could not add basemap: {str(e)}")
                # Elegant fallback - light gray background with subtle pattern
                ax.set_facecolor('#f8f9fa')
                # Add a subtle grid pattern as background
                ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5, color='#34495e')
            
            # Set title and labels
            model_display_name = self.display_mapper.get_display_name(model_name)
            ax.set_title(f'{model_display_name} Model Results', fontsize=16, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add north arrow and scale bar if possible
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Tight layout
            plt.tight_layout()
            
            # Save map
            map_filename = f"map_{model_name}_{self.analysis.request_id}.png"
            map_path = self.temp_dir / map_filename
            plt.savefig(map_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return map_path
            
        except Exception as e:
            logger.error(f"Error generating map for {model_name}: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory: {str(e)}")