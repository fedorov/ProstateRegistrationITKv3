/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    DeformableRegistration15.cxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

// Software Guide : BeginLatex
//
// This example illustrates a realistic pipeline for solving a full deformable registration problem.
//
// First the two images are roughly aligned by using a transform
// initialization, then they are registered using a rigid transform, that in
// turn, is used to initialize a registration with an affine transform. The
// transform resulting from the affine registration is used as the bulk
// transform of a BSplineDeformableTransform. The deformable registration is
// computed, and finally the resulting transform is used to resample the moving
// image.
//
// Software Guide : EndLatex

#include "itkImageRegistrationMethod.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkOrientedImage.h"

#include "itkTimeProbesCollectorBase.h"

#ifdef ITK_USE_REVIEW
#include "itkMemoryProbesCollectorBase.h"
#define itkProbesCreate()  \
  itk::TimeProbesCollectorBase chronometer; \
  itk::MemoryProbesCollectorBase memorymeter
#define itkProbesStart( text ) memorymeter.Start( text ); chronometer.Start( text )
#define itkProbesStop( text )  chronometer.Stop( text ); memorymeter.Stop( text  )
#define itkProbesReport( stream )  chronometer.Report( stream ); memorymeter.Report( stream  )
#else
#define itkProbesCreate()  \
  itk::TimeProbesCollectorBase chronometer
#define itkProbesStart( text ) chronometer.Start( text )
#define itkProbesStop( text )  chronometer.Stop( text )
#define itkProbesReport( stream )  chronometer.Report( stream )
#endif

//  Software Guide : BeginLatex
//
//  The following are the most relevant headers to this example.
//
//  \index{itk::VersorRigid3DTransform!header}
//  \index{itk::AffineTransform!header}
//  \index{itk::BSplineDeformableTransform!header}
//  \index{itk::RegularStepGradientDescentOptimizer!header}
//
//  Software Guide : EndLatex

// Software Guide : BeginCodeSnippet
#include "itkCenteredTransformInitializer.h"
#include "itkVersorRigid3DTransform.h"
#include "itkEuler3DTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineDeformableTransform.h"
#include "itkRegularStepGradientDescentOptimizer.h"
// Software Guide : EndCodeSnippet

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"


#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "itkImageMaskSpatialObject.h"
//#include "itkLabelMap.h"
#include "itkStatisticsLabelObject.h"
#include "itkLabelImageToStatisticsLabelMapFilter.h"

//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"
class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:
  typedef itk::RegularStepGradientDescentOptimizer  OptimizerType;
  typedef   const OptimizerType *                   OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    OptimizerPointer optimizer =
      dynamic_cast< OptimizerPointer >( object );
    if( !(itk::IterationEvent().CheckEvent( &event )) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << std::endl;
    }
};

int main( int argc, char *argv[] )
{
  if( argc < 5 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << " fixedImageMaskFile movingImageMaskFile ";
    std::cerr << " [differenceOutputfile] [differenceBeforeRegistration] ";
    std::cerr << " [deformationField] ";
    std::cerr << " [useExplicitPDFderivatives ] [useCachingBSplineWeights ] ";
    std::cerr << " [filenameForFinalTransformParameters] ";
    std::cerr << " [numberOfGridNodesInsideImageInOneDimensionCoarse] ";
    std::cerr << " [numberOfGridNodesInsideImageInOneDimensionFine] ";
    std::cerr << " [maximumStepLength] [maximumNumberOfIterations]";
    std::cerr << std::endl;
    return EXIT_FAILURE;
    }

  char *fixedImageFile = argv[1],
    *movingImageFile = argv[2],
    *outputImageFile = argv[100],
    *fixedImageMaskFile = argv[3],
    *movingImageMaskFile = argv[4];

  const    unsigned int    ImageDimension = 3;
  typedef  signed short    PixelType;
  typedef unsigned char    MaskPixelType;

  typedef itk::OrientedImage< PixelType, ImageDimension >  FixedImageType;
  typedef itk::OrientedImage< PixelType, ImageDimension >  MovingImageType;

  typedef itk::OrientedImage< MaskPixelType, ImageDimension >  MaskImageType;

  typedef itk::ImageMaskSpatialObject<3> MaskType;
  typedef MaskType::Pointer MaskPointer;

  const unsigned int SpaceDimension = ImageDimension;

  const unsigned int SplineOrder = 3;
  typedef double CoordinateRepType;

  typedef itk::VersorRigid3DTransform< double > RigidTransformType;

  typedef itk::AffineTransform< double, SpaceDimension > AffineTransformType;

  typedef itk::BSplineDeformableTransform<
                            CoordinateRepType,
                            SpaceDimension,
                            SplineOrder >     DeformableTransformType;

  typedef itk::CenteredTransformInitializer< RigidTransformType,
                                             FixedImageType,
                                             MovingImageType
                                                 >  TransformInitializerType;


  typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;


  typedef itk::MattesMutualInformationImageToImageMetric<
                                    FixedImageType,
                                    MovingImageType >    MetricType;

  typedef itk:: LinearInterpolateImageFunction<
                                    MovingImageType,
                                    double          >    InterpolatorType;

  typedef itk::ImageRegistrationMethod<
                                    FixedImageType,
                                    MovingImageType >    RegistrationType;

  MetricType::Pointer         metric        = MetricType::New();
  OptimizerType::Pointer      optimizer     = OptimizerType::New();
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  RegistrationType::Pointer   registration  = RegistrationType::New();


  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetInterpolator(  interpolator  );

  // Auxiliary identity transform.
  typedef itk::IdentityTransform<double,SpaceDimension> IdentityTransformType;
  IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();


  //
  //   Read the Fixed and Moving images.
  //
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;
  typedef itk::ImageFileReader< MaskImageType > MaskImageReaderType;

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(  fixedImageFile );
  movingImageReader->SetFileName( movingImageFile );

  try
    {
    fixedImageReader->Update();
    movingImageReader->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  MovingImageType::Pointer movingImage = movingImageReader->GetOutput();
  MaskImageReaderType::Pointer  fixedImageMaskReader  = MaskImageReaderType::New();
  MaskImageReaderType::Pointer movingImageMaskReader = MaskImageReaderType::New();

  fixedImageMaskReader->SetFileName(  fixedImageMaskFile );
  movingImageMaskReader->SetFileName( movingImageMaskFile );

  try
    {
    fixedImageMaskReader->Update();
    movingImageMaskReader->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  MaskImageType::Pointer fixedImageMask = fixedImageMaskReader->GetOutput();
  MaskImageType::Pointer movingImageMask = movingImageMaskReader->GetOutput();


  MaskPointer fixedMask = MaskType::New();
  fixedMask->SetImage(fixedImageMask);
  fixedMask->ComputeObjectToWorldTransform();

  MaskPointer movingMask = MaskType::New();
  movingMask->SetImage(movingImageMask);
  movingMask->ComputeObjectToWorldTransform();

  //registration->SetFixedImage(  fixedImage   );
  //registration->SetMovingImage(   movingImageReader->GetOutput()   );

  // BRAINSFitHelper.cxx:116
  metric->SetFixedImage(  fixedImage  );
  metric->SetMovingImage(   movingImage   );

  metric->SetFixedImageMask(  fixedMask   );
  metric->SetMovingImageMask(   movingMask   );

  metric->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion());
  metric->ReinitializeSeed(76926294);
  metric->SetNumberOfSpatialSamples(10000);
  metric->SetNumberOfHistogramBins( 50 );
  metric->SetUseCachingOfBSplineWeights(1);
  metric->SetUseExplicitPDFDerivatives(1);

  // Calculate the initial transform by first aligning the centers of the masks,
  // and then doing sparse search
  typename MovingImageType::PointType movingCenter;
  typename FixedImageType::PointType fixedCenter;

  // calculate the centers of each ROI
  typedef itk::StatisticsLabelObject< unsigned char, 3 > LabelObjectType;
  typedef itk::LabelMap< LabelObjectType >               LabelMapType;
  typedef itk::LabelImageToStatisticsLabelMapFilter< MaskImageType, MaskImageType >
  LabelStatisticsFilterType;
  typedef LabelMapType::LabelObjectContainerType LabelObjectContainerType;

  LabelStatisticsFilterType::Pointer movingImageToLabel = LabelStatisticsFilterType::New();
  movingImageToLabel->SetInput( movingImageMask );
  movingImageToLabel->SetFeatureImage(movingImageMask );
  movingImageToLabel->SetComputePerimeter(false);
  movingImageToLabel->Update();

  LabelStatisticsFilterType::Pointer fixedImageToLabel = LabelStatisticsFilterType::New();
  fixedImageToLabel->SetInput( fixedImageMask );
  fixedImageToLabel->SetFeatureImage( fixedImageMask );
  fixedImageToLabel->SetComputePerimeter(false);
  fixedImageToLabel->Update();

  LabelObjectType *movingLabel = movingImageToLabel->GetOutput()->GetNthLabelObject(0);
  LabelObjectType *fixedLabel = fixedImageToLabel->GetOutput()->GetNthLabelObject(0);

  LabelObjectType::CentroidType movingCentroid = movingLabel->GetCentroid();
  LabelObjectType::CentroidType fixedCentroid = fixedLabel->GetCentroid();

  movingCenter[0] = movingCentroid[0];
  movingCenter[1] = movingCentroid[1];
  movingCenter[2] = movingCentroid[2];

  fixedCenter[0] = fixedCentroid[0];
  fixedCenter[1] = fixedCentroid[1];
  fixedCenter[2] = fixedCentroid[2];

  // do the search
  RigidTransformType::InputPointType rotationCenter;
  RigidTransformType::OutputVectorType translationVector;
  itk::Vector< double, 3 > scaleValue;

  for ( unsigned int i = 0; i < 3; i++ )
    {
    rotationCenter[i]    = fixedCenter[i];
    translationVector[i] = movingCenter[i] - fixedCenter[i];
    scaleValue[i] = 1;
    }
  typedef itk::Euler3DTransform< double > EulerAngle3DTransformType;
  EulerAngle3DTransformType::Pointer bestEulerAngles3D = EulerAngle3DTransformType::New();
  bestEulerAngles3D->SetCenter(rotationCenter);
  bestEulerAngles3D->SetTranslation(translationVector);

  typedef itk::Euler3DTransform< double > EulerAngle3DTransformType;
  EulerAngle3DTransformType::Pointer currentEulerAngles3D = EulerAngle3DTransformType::New();
  currentEulerAngles3D->SetCenter(rotationCenter);
  currentEulerAngles3D->SetTranslation(translationVector);

  /*
  itk::TransformFileWriter::Pointer tw = itk::TransformFileWriter::New();
  tw->SetInput(currentEulerAngles3D);
  tw->SetFileName("/tmp/e.tfm");
  tw->Update();
  */

  metric->SetTransform(currentEulerAngles3D);
  metric->Initialize();
  double max_cc = 0.0;
  // void QuickSampleParameterSpace(void)
    {
    currentEulerAngles3D->SetRotation(0, 0, 0);
    // Initialize with current guess;
    max_cc = metric->GetValue( currentEulerAngles3D->GetParameters() );
    const double HARange = 12.0;
    const double PARange = 12.0;
    // rough search in neighborhood.
    const double one_degree = 1.0F * vnl_math::pi / 180.0F;
    const double HAStepSize = 3.0 * one_degree;
    const double PAStepSize = 3.0 * one_degree;
    // Quick search just needs to get an approximate angle correct.
      {
      for ( double HA = -HARange * one_degree; HA <= HARange * one_degree; HA += HAStepSize )
        {
        for ( double PA = -PARange * one_degree; PA <= PARange * one_degree; PA += PAStepSize )
          {
          currentEulerAngles3D->SetRotation(PA, 0, HA);
          const double current_cc = metric->GetValue( currentEulerAngles3D->GetParameters() );
          if ( current_cc < max_cc )
            {
            max_cc = current_cc;
            bestEulerAngles3D->SetFixedParameters( currentEulerAngles3D->GetFixedParameters() );
            bestEulerAngles3D->SetParameters( currentEulerAngles3D->GetParameters() );
            }
          // #define DEBUGGING_PRINT_IMAGES

          }
        }
      }
    }

  RigidTransformType::Pointer quickSetVersor = RigidTransformType::New();
  quickSetVersor->SetCenter( bestEulerAngles3D->GetCenter() );
  quickSetVersor->SetTranslation( bestEulerAngles3D->GetTranslation() );
  {
    itk::Versor< double > localRotation;
    localRotation.Set( bestEulerAngles3D->GetRotationMatrix() );
    quickSetVersor->SetRotation(localRotation);
  }

  /*
  itk::TransformFileWriter::Pointer tw = itk::TransformFileWriter::New();
  tw->SetInput(quickSetVersor);
  tw->SetFileName("/tmp/e.tfm");
  tw->Update();
  */

  //
  // Add a time and memory probes collector for profiling the computation time
  // of every stage.
  //
  //itkProbesCreate();

  registration->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion() );
  registration->SetInitialTransformParameters( quickSetVersor->GetParameters() );
  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);

  RigidTransformType::Pointer rigidTransform = RigidTransformType::New();

  registration->SetTransform( rigidTransform );

  //
  //  Define optimizer normaliztion to compensate for different dynamic range
  //  of rotations and translations.
  //
  typedef OptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( rigidTransform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 1000.0;
  const double reproportionScale = 1.0 / 1.0;
  const double skewScale = 1.0 / 1.0;

  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;

  optimizer->SetScales( optimizerScales );

  // BRAINSFitHelperTemplate.cxx:521
  optimizer->SetMaximumStepLength( 0.2000  );
  optimizer->SetMinimumStepLength( 0.005 );

  optimizer->SetNumberOfIterations( 1500 );

  //
  // The rigid transform has 6 parameters we use therefore a few samples to run
  // this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  metric->SetNumberOfSpatialSamples( 500000 );

  //
  // Create the Command observer and register it with the optimizer.
  //
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );


  std::cout << "Starting Rigid Registration " << std::endl;

  try
    {
    //itkProbesStart( "Rigid Registration" );
    registration->StartRegistration();
    //itkProbesStop( "Rigid Registration" );
    std::cout << "Optimizer stop condition = "
              << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Rigid Registration completed" << std::endl;
  std::cout << std::endl;

  rigidTransform->SetParameters( registration->GetLastTransformParameters() );

  //
  //  Perform Affine Registration
  //
  AffineTransformType::Pointer  affineTransform = AffineTransformType::New();

  affineTransform->SetCenter( rigidTransform->GetCenter() );
  affineTransform->SetTranslation( rigidTransform->GetTranslation() );
  affineTransform->SetMatrix( rigidTransform->GetMatrix() );

  registration->SetTransform( affineTransform );
  registration->SetInitialTransformParameters( affineTransform->GetParameters() );

  optimizerScales = OptimizerScalesType( affineTransform->GetNumberOfParameters() );

  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = 1.0;
  optimizerScales[4] = 1.0;
  optimizerScales[5] = 1.0;
  optimizerScales[6] = 1.0;
  optimizerScales[7] = 1.0;
  optimizerScales[8] = 1.0;

  optimizerScales[9]  = translationScale;
  optimizerScales[10] = translationScale;
  optimizerScales[11] = translationScale;

  optimizer->SetScales( optimizerScales );

  //optimizer->SetMaximumStepLength( 0.2000  );
  //optimizer->SetMinimumStepLength( 0.0001 );

  //optimizer->SetNumberOfIterations( 200 );

  //
  // The Affine transform has 12 parameters we use therefore a more samples to run
  // this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  //metric->SetNumberOfSpatialSamples( 500000L );


  std::cout << "Starting Affine Registration " << std::endl;

  try
    {
    //itkProbesStart( "Affine Registration" );
    registration->StartRegistration();
    //itkProbesStop( "Affine Registration" );
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Affine Registration completed" << std::endl;
  std::cout << std::endl;

  affineTransform->SetParameters( registration->GetLastTransformParameters() );

  return 0;
}
#if 0
  //
  //  Perform Deformable Registration
  //
  DeformableTransformType::Pointer  bsplineTransformCoarse = DeformableTransformType::New();

  unsigned int numberOfGridNodesInOneDimensionCoarse = 5;

  if( argc > 10 )
    {
    numberOfGridNodesInOneDimensionCoarse = atoi( argv[10] );
    }


  typedef DeformableTransformType::RegionType RegionType;
  RegionType bsplineRegion;
  RegionType::SizeType   gridSizeOnImage;
  RegionType::SizeType   gridBorderSize;
  RegionType::SizeType   totalGridSize;

  gridSizeOnImage.Fill( numberOfGridNodesInOneDimensionCoarse );
  gridBorderSize.Fill( SplineOrder );    // Border for spline order = 3 ( 1 lower, 2 upper )
  totalGridSize = gridSizeOnImage + gridBorderSize;

  bsplineRegion.SetSize( totalGridSize );

  typedef DeformableTransformType::SpacingType SpacingType;
  SpacingType spacing = fixedImage->GetSpacing();

  typedef DeformableTransformType::OriginType OriginType;
  OriginType origin = fixedImage->GetOrigin();

  FixedImageType::SizeType fixedImageSize = fixedRegion.GetSize();

  for(unsigned int r=0; r<ImageDimension; r++)
    {
    spacing[r] *= static_cast<double>(fixedImageSize[r] - 1)  /
                  static_cast<double>(gridSizeOnImage[r] - 1);
    }

  FixedImageType::DirectionType gridDirection = fixedImage->GetDirection();
  SpacingType gridOriginOffset = gridDirection * spacing;

  OriginType gridOrigin = origin - gridOriginOffset;

  bsplineTransformCoarse->SetGridSpacing( spacing );
  bsplineTransformCoarse->SetGridOrigin( gridOrigin );
  bsplineTransformCoarse->SetGridRegion( bsplineRegion );
  bsplineTransformCoarse->SetGridDirection( gridDirection );

  bsplineTransformCoarse->SetBulkTransform( affineTransform );

  typedef DeformableTransformType::ParametersType     ParametersType;

  unsigned int numberOfBSplineParameters = bsplineTransformCoarse->GetNumberOfParameters();


  optimizerScales = OptimizerScalesType( numberOfBSplineParameters );
  optimizerScales.Fill( 1.0 );

  optimizer->SetScales( optimizerScales );


  ParametersType initialDeformableTransformParameters( numberOfBSplineParameters );
  initialDeformableTransformParameters.Fill( 0.0 );

  bsplineTransformCoarse->SetParameters( initialDeformableTransformParameters );

  registration->SetInitialTransformParameters( bsplineTransformCoarse->GetParameters() );
  registration->SetTransform( bsplineTransformCoarse );

  // Software Guide : EndCodeSnippet


  //  Software Guide : BeginLatex
  //
  //  Next we set the parameters of the RegularStepGradientDescentOptimizer object.
  //
  //  Software Guide : EndLatex


  // Software Guide : BeginCodeSnippet
  optimizer->SetMaximumStepLength( 10.0 );
  optimizer->SetMinimumStepLength(  0.01 );

  optimizer->SetRelaxationFactor( 0.7 );
  optimizer->SetNumberOfIterations( 50 );
  // Software Guide : EndCodeSnippet


  // Optionally, get the step length from the command line arguments
  if( argc > 11 )
    {
    optimizer->SetMaximumStepLength( atof( argv[12] ) );
    }

  // Optionally, get the number of iterations from the command line arguments
  if( argc > 12 )
    {
    optimizer->SetNumberOfIterations( atoi( argv[13] ) );
    }


  //
  // The BSpline transform has a large number of parameters, we use therefore a
  // much larger number of samples to run this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  metric->SetNumberOfSpatialSamples( numberOfBSplineParameters * 100 );

  std::cout << std::endl << "Starting Deformable Registration Coarse Grid" << std::endl;

  try
    {
    itkProbesStart( "Deformable Registration Coarse" );
    registration->StartRegistration();
    itkProbesStop( "Deformable Registration Coarse" );
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Deformable Registration Coarse Grid completed" << std::endl;
  std::cout << std::endl;

  OptimizerType::ParametersType finalParameters =
                    registration->GetLastTransformParameters();

  bsplineTransformCoarse->SetParameters( finalParameters );

  //  Software Guide : BeginLatex
  //
  //  Once the registration has finished with the low resolution grid, we
  //  proceed to instantiate a higher resolution
  //  \code{BSplineDeformableTransform}.
  //
  //  Software Guide : EndLatex

  DeformableTransformType::Pointer  bsplineTransformFine = DeformableTransformType::New();

  unsigned int numberOfGridNodesInOneDimensionFine = 20;

  if( argc > 11 )
    {
    numberOfGridNodesInOneDimensionFine = atoi( argv[11] );
    }

  RegionType::SizeType   gridHighSizeOnImage;
  gridHighSizeOnImage.Fill( numberOfGridNodesInOneDimensionFine );
  totalGridSize = gridHighSizeOnImage + gridBorderSize;

  bsplineRegion.SetSize( totalGridSize );

  SpacingType spacingHigh = fixedImage->GetSpacing();
  OriginType  originHigh  = fixedImage->GetOrigin();

  for(unsigned int rh=0; rh<ImageDimension; rh++)
    {
    spacingHigh[rh] *= static_cast<double>(fixedImageSize[rh] - 1)  /
      static_cast<double>(gridHighSizeOnImage[rh] - 1);
    originHigh[rh] -= spacingHigh[rh];
    }

  SpacingType gridOriginOffsetHigh = gridDirection * spacingHigh;

  OriginType gridOriginHigh = origin - gridOriginOffsetHigh;


  bsplineTransformFine->SetGridSpacing( spacingHigh );
  bsplineTransformFine->SetGridOrigin( gridOriginHigh );
  bsplineTransformFine->SetGridRegion( bsplineRegion );
  bsplineTransformFine->SetGridDirection( gridDirection );

  bsplineTransformFine->SetBulkTransform( affineTransform );

  numberOfBSplineParameters = bsplineTransformFine->GetNumberOfParameters();

  ParametersType parametersHigh( numberOfBSplineParameters );
  parametersHigh.Fill( 0.0 );

  //  Software Guide : BeginLatex
  //
  //  Now we need to initialize the BSpline coefficients of the higher resolution
  //  transform. This is done by first computing the actual deformation field
  //  at the higher resolution from the lower resolution BSpline coefficients.
  //  Then a BSpline decomposition is done to obtain the BSpline coefficient of
  //  the higher resolution transform.
  //
  //  Software Guide : EndLatex

  unsigned int counter = 0;

  for ( unsigned int k = 0; k < SpaceDimension; k++ )
    {
    typedef DeformableTransformType::ImageType ParametersImageType;
    typedef itk::ResampleImageFilter<ParametersImageType,ParametersImageType> ResamplerType;
    ResamplerType::Pointer upsampler = ResamplerType::New();

    typedef itk::BSplineResampleImageFunction<ParametersImageType,double> FunctionType;
    FunctionType::Pointer function = FunctionType::New();

    upsampler->SetInput( bsplineTransformCoarse->GetCoefficientImage()[k] );
    upsampler->SetInterpolator( function );
    upsampler->SetTransform( identityTransform );
    upsampler->SetSize( bsplineTransformFine->GetGridRegion().GetSize() );
    upsampler->SetOutputSpacing( bsplineTransformFine->GetGridSpacing() );
    upsampler->SetOutputOrigin( bsplineTransformFine->GetGridOrigin() );

    typedef itk::BSplineDecompositionImageFilter<ParametersImageType,ParametersImageType>
      DecompositionType;
    DecompositionType::Pointer decomposition = DecompositionType::New();

    decomposition->SetSplineOrder( SplineOrder );
    decomposition->SetInput( upsampler->GetOutput() );
    decomposition->Update();

    ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();

    // copy the coefficients into the parameter array
    typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
    Iterator it( newCoefficients, bsplineTransformFine->GetGridRegion() );
    while ( !it.IsAtEnd() )
      {
      parametersHigh[ counter++ ] = it.Get();
      ++it;
      }

    }


  optimizerScales = OptimizerScalesType( numberOfBSplineParameters );
  optimizerScales.Fill( 1.0 );

  optimizer->SetScales( optimizerScales );

  bsplineTransformFine->SetParameters( parametersHigh );

  //  Software Guide : BeginLatex
  //
  //  We now pass the parameters of the high resolution transform as the initial
  //  parameters to be used in a second stage of the registration process.
  //
  //  Software Guide : EndLatex

  std::cout << "Starting Registration with high resolution transform" << std::endl;

  // Software Guide : BeginCodeSnippet
  registration->SetInitialTransformParameters( bsplineTransformFine->GetParameters() );
  registration->SetTransform( bsplineTransformFine );

  //
  // The BSpline transform at fine scale has a very large number of parameters,
  // we use therefore a much larger number of samples to run this stage. In this
  // case, however, the number of transform parameters is closer to the number
  // of pixels in the image. Therefore we use the geometric mean of the two numbers
  // to ensure that the number of samples is larger than the number of transform
  // parameters and smaller than the number of samples.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  const unsigned long numberOfSamples =
     static_cast<unsigned long>(
       vcl_sqrt( static_cast<double>( numberOfBSplineParameters ) *
                 static_cast<double>( numberOfPixels ) ) );

  metric->SetNumberOfSpatialSamples( numberOfSamples );


  try
    {
    itkProbesStart( "Deformable Registration Fine" );
    registration->StartRegistration();
    itkProbesStop( "Deformable Registration Fine" );
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  // Software Guide : EndCodeSnippet


  std::cout << "Deformable Registration Fine Grid completed" << std::endl;
  std::cout << std::endl;


  // Report the time and memory taken by the registration
  itkProbesReport( std::cout );

  finalParameters = registration->GetLastTransformParameters();

  bsplineTransformFine->SetParameters( finalParameters );


  typedef itk::ResampleImageFilter<
                            MovingImageType,
                            FixedImageType >    ResampleFilterType;

  ResampleFilterType::Pointer resample = ResampleFilterType::New();

  resample->SetTransform( bsplineTransformFine );
  resample->SetInput( movingImageReader->GetOutput() );

  resample->SetSize(    fixedImage->GetLargestPossibleRegion().GetSize() );
  resample->SetOutputOrigin(  fixedImage->GetOrigin() );
  resample->SetOutputSpacing( fixedImage->GetSpacing() );
  resample->SetOutputDirection( fixedImage->GetDirection() );

  // This value is set to zero in order to make easier to perform
  // regression testing in this example. However, for didactic
  // exercise it will be better to set it to a medium gray value
  // such as 100 or 128.
  resample->SetDefaultPixelValue( 0 );

  typedef  signed short  OutputPixelType;

  typedef itk::Image< OutputPixelType, ImageDimension > OutputImageType;

  typedef itk::CastImageFilter<
                        FixedImageType,
                        OutputImageType > CastFilterType;

  typedef itk::ImageFileWriter< OutputImageType >  WriterType;


  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();


  writer->SetFileName( argv[3] );


  caster->SetInput( resample->GetOutput() );
  writer->SetInput( caster->GetOutput()   );

  std::cout << "Writing resampled moving image...";

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << " Done!" << std::endl;


  typedef itk::SquaredDifferenceImageFilter<
                                  FixedImageType,
                                  FixedImageType,
                                  OutputImageType > DifferenceFilterType;

  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();

  WriterType::Pointer writer2 = WriterType::New();
  writer2->SetInput( difference->GetOutput() );


  // Compute the difference image between the
  // fixed and resampled moving image.
  if( argc > 4 )
    {
    difference->SetInput1( fixedImageReader->GetOutput() );
    difference->SetInput2( resample->GetOutput() );
    writer2->SetFileName( argv[4] );

    std::cout << "Writing difference image after registration...";

    try
      {
      writer2->Update();
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }

    std::cout << " Done!" << std::endl;
    }


  // Compute the difference image between the
  // fixed and moving image before registration.
  if( argc > 5 )
    {
    writer2->SetFileName( argv[5] );
    difference->SetInput1( fixedImageReader->GetOutput() );
    resample->SetTransform( identityTransform );

    std::cout << "Writing difference image before registration...";

    try
      {
      writer2->Update();
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }

    std::cout << " Done!" << std::endl;
    }

  // Generate the explicit deformation field resulting from
  // the registration.
  if( argc > 6 )
    {

    typedef itk::Vector< float, ImageDimension >      VectorType;
    typedef itk::Image< VectorType, ImageDimension >  DeformationFieldType;

    DeformationFieldType::Pointer field = DeformationFieldType::New();
    field->SetRegions( fixedRegion );
    field->SetOrigin( fixedImage->GetOrigin() );
    field->SetSpacing( fixedImage->GetSpacing() );
    field->SetDirection( fixedImage->GetDirection() );
    field->Allocate();

    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi( field, fixedRegion );

    fi.GoToBegin();

    DeformableTransformType::InputPointType  fixedPoint;
    DeformableTransformType::OutputPointType movingPoint;
    DeformationFieldType::IndexType index;

    VectorType displacement;

    while( ! fi.IsAtEnd() )
      {
      index = fi.GetIndex();
      field->TransformIndexToPhysicalPoint( index, fixedPoint );
      movingPoint = bsplineTransformFine->TransformPoint( fixedPoint );
      displacement = movingPoint - fixedPoint;
      fi.Set( displacement );
      ++fi;
      }

    typedef itk::ImageFileWriter< DeformationFieldType >  FieldWriterType;
    FieldWriterType::Pointer fieldWriter = FieldWriterType::New();

    fieldWriter->SetInput( field );

    fieldWriter->SetFileName( argv[6] );

    std::cout << "Writing deformation field ...";

    try
      {
      fieldWriter->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
      std::cerr << "Exception thrown " << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
      }

    std::cout << " Done!" << std::endl;
    }

  // Optionally, save the transform parameters in a file
  if( argc > 9 )
    {
    std::cout << "Writing transform parameter file ...";
    std::ofstream parametersFile;
    parametersFile.open( argv[9] );
    parametersFile << finalParameters << std::endl;
    parametersFile.close();
    std::cout << " Done!" << std::endl;
    }

  return EXIT_SUCCESS;
}

#undef itkProbesCreate
#undef itkProbesStart
#undef itkProbesStop
#undef itkProbesReport
#endif // 0
