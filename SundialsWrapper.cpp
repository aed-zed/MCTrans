
#include "SundialsWrapper.hpp"

void ArkodeErrorWrapper( int errorFlag, std::string&& fName )
{
	if ( errorFlag == ARK_SUCCESS )
		return;
	else
	{
		std::string errName = ARKodeGetReturnFlagName( errorFlag );
		throw std::runtime_error( "Error " + errName + " returned from ARKode function: " + fName );
	}
}

int ARKode_TemperatureSolve( realtype t, N_Vector u, N_Vector uDot, void* voidPlasma )
{
	MirrorPlasma* plasmaPtr = reinterpret_cast<MirrorPlasma*>( voidPlasma );

	double TiOld = plasmaPtr->IonTemperature;
	double TeOld = plasmaPtr->ElectronTemperature;

	if ( ION_TEMPERATURE( u ) < 0.0 ) {
#if defined( DEBUG )
		std::cerr << "Error in SUNDIALS solve, due to negative ion temperature" << std::endl;
#endif
		return 1;
	}
	if ( ELECTRON_TEMPERATURE( u ) < 0.0 ) {
#if defined( DEBUG )
		std::cerr << "Error in SUNDIALS solve, due to negative electron temperature" << std::endl;
#endif
		return 2;
	}

	plasmaPtr->IonTemperature = ION_TEMPERATURE( u );
	plasmaPtr->ElectronTemperature = ELECTRON_TEMPERATURE( u );
	//plasma->ElectronDensity = DENSITY( u );
	//plasma->SetIonDensity() // Set n_i from n_e, Z_i
	
	try {
		plasmaPtr->SetTime(t);
	} catch ( std::domain_error &e ) {
		// Timestep too long?
#ifdef DEBUG
		std::cerr << "Evaluating RHS at t = " << std::setprecision( 20 ) << t << " ?!" << std::endl;
#endif
		return 3;
	}
	plasmaPtr->SetMachFromVoltage();
	plasmaPtr->UpdatePhi();
	plasmaPtr->ComputeSteadyStateNeutrals();
#if defined( DEBUG ) && defined( SUNDIALS_DEBUG ) && defined( INTERNAL_RK_DEBUG )
	std::cerr << "t = " << t << " ; T_i = " << plasmaPtr->IonTemperature << " ; T_e = " << plasmaPtr->ElectronTemperature << " MachNumber " << plasmaPtr->MachNumber << std::endl;
#endif


	try {
		double IonHeating  = plasmaPtr->IonHeating();
		double IonHeatLoss = plasmaPtr->IonHeatLosses();
		double ElectronHeating  = plasmaPtr->ElectronHeating();
		double ElectronHeatLoss = plasmaPtr->ElectronHeatLosses();

#if defined( DEBUG ) && defined( SUNDIALS_DEBUG ) && defined( INTERNAL_RK_DEBUG )
		std::cerr << " Ion Heating      = " << IonHeating      << " ; Ion Heat Loss       = " << IonHeatLoss      << std::endl;
		std::cerr << " Electron Heating = " << ElectronHeating << " ; Electron Heat Loss  = " << ElectronHeatLoss << std::endl;
#endif

		ION_HEAT_BALANCE( uDot )      = ( IonHeating - IonHeatLoss );
		ELECTRON_HEAT_BALANCE( uDot ) = ( ElectronHeating - ElectronHeatLoss );
//		PARTICLE_BALANCE( uDot ) = ParticleBalance; 


	} catch ( std::exception& e ) {
		return -1;
	} 

	plasmaPtr->IonTemperature = TiOld;
	plasmaPtr->ElectronTemperature = TeOld;
	plasmaPtr->SetMachFromVoltage();
	plasmaPtr->UpdatePhi();
	plasmaPtr->ComputeSteadyStateNeutrals();

	return ARK_SUCCESS;
}

// In this mode, Mach Number is u(0) and T_i is u(1)
// but we still solve both the power-balance equations by running to steady state
int ARKode_FixedTeSolve( realtype t, N_Vector u, N_Vector F, void* voidPlasma )
{
	MirrorPlasma* plasmaPtr = reinterpret_cast<MirrorPlasma*>( voidPlasma );

	plasmaPtr->MachNumber = MACH_NUMBER( u );

	plasmaPtr->IonTemperature = ION_TEMPERATURE( u );
	

	try {
		double IonHeating  = plasmaPtr->IonHeating();
		double IonHeatLoss = plasmaPtr->IonHeatLosses();
		double ElectronHeating  = plasmaPtr->ElectronHeating();
		double ElectronHeatLoss = plasmaPtr->ElectronHeatLosses();

		ION_HEAT_BALANCE( F )      = ( IonHeating - IonHeatLoss );
		ELECTRON_HEAT_BALANCE( F ) = ( ElectronHeating - ElectronHeatLoss );
//		PARTICLE_BALANCE( F ) = ParticleBalance; 

	} catch ( std::exception& e ) {
		return -1;
	} 

	return 0;
}


void MCTransConfig::doTempSolve( MirrorPlasma& plasma ) const
{
	sundials::Context sunctx;
	sunindextype NDims = N_DIMENSIONS;
	N_Vector initialCondition = N_VNew_Serial( NDims, sunctx );

	ION_TEMPERATURE( initialCondition ) = plasma.IonTemperature;
	ELECTRON_TEMPERATURE( initialCondition ) = plasma.ElectronTemperature;
	
	realtype t0 = 0;

	plasma.SetTime( t0 );
	plasma.SetMachFromVoltage();
	plasma.UpdatePhi();
	plasma.ComputeSteadyStateNeutrals();


	void *arkMem = ARKStepCreate( nullptr, ARKode_TemperatureSolve, t0, initialCondition, sunctx );

	if ( arkMem == nullptr ) {
		throw std::runtime_error( "Cannot allocate ARKode Working Memory" );
	}

	// Dummy Jacobian, will be filled by ARKode with finite-difference approximations
	SUNMatrix       Jacobian = SUNDenseMatrix( NDims, NDims, sunctx );
	// Small system, direct solve is fastest
	SUNLinearSolver  LS = SUNLinSol_Dense( initialCondition, Jacobian, sunctx );

	ArkodeErrorWrapper( ARKodeSetLinearSolver( arkMem, LS, Jacobian ), "ARKodeSetLinearSolver" );

	double abstol = plasma.SundialsAbsTol;
	double reltol = plasma.SundialsRelTol;

#ifdef DEBUG
	std::cerr << "Using SundialsAbsTol = " << abstol << " and SundialsRelTol = " << reltol << std::endl;
#endif
	ArkodeErrorWrapper( ARKodeSStolerances( arkMem, reltol, abstol ), "ARKodeSStolerances" );

	ArkodeErrorWrapper( ARKStepSetTableNum( arkMem, IRK_SCHEME, ARKSTEP_NULL_STEPPER ), "ARKodeSetTableNum" );

	ArkodeErrorWrapper( ARKodeSetUserData( arkMem, reinterpret_cast<void*>( &plasma ) ), "ARKodeSetUserData" );


	N_Vector positivityEnforcement = N_VNew_Serial( NDims, sunctx );
	N_VConst( 0.0, positivityEnforcement ); // Default to no constraints
	ION_TEMPERATURE( positivityEnforcement ) = 2.0;      // T_i > 0
	ELECTRON_TEMPERATURE( positivityEnforcement ) = 2.0; // T_e > 0

	ArkodeErrorWrapper( ARKodeSetConstraints( arkMem, positivityEnforcement ), "ARKodeSetConstraints" );

	// Because the scheme is 4th order, we request cubic hermite interpolation between
	// internal timesteps, and don't allow the timestep to exceed 5*dt where dt is the
	// time between outputs.

	ArkodeErrorWrapper( ARKodeSetInterpolantDegree( arkMem, 3 ), "ARKodeSetInterpolantDegree" );
	ArkodeErrorWrapper( ARKodeSetMaxStep( arkMem, OutputDeltaT*5 ), "ARKodeSetMaxStep" );

	const unsigned long MaxSteps = 1e5;
	ArkodeErrorWrapper( ARKodeSetMaxNumSteps( arkMem, MaxSteps ), "ARKodeSetMaxNumSteps" );

	realtype t,tRet = 0;	
	int errorFlag;

#ifdef DEBUG
	std::cerr << "Solving from t = " << plasma.time << " to t = " << EndTime << std::endl;
	std::cerr << "Writing output every " << OutputDeltaT << std::endl;
#endif 
	ArkodeErrorWrapper( ARKodeSetStopTime( arkMem, EndTime ), "ARKodeSetStopTime" );
	for ( t = OutputDeltaT; t < EndTime; t += OutputDeltaT )
	{
#if defined( DEBUG )
		double curTime;
		ArkodeErrorWrapper( ARKodeGetCurrentTime( arkMem, &curTime ), "ARKodeGetCurrentTime" );
#endif
		if ( t > EndTime )
			t = EndTime;
		errorFlag = ARKodeEvolve( arkMem, t, initialCondition, &tRet, ARK_NORMAL );
		switch ( errorFlag ) {
			case ARK_SUCCESS:
#if defined( DEBUG )
				std::cerr << "Internal time is " << curTime << " Evolved to " << tRet << " with intent of reaching " << t << std::endl;
#endif
				break;
			default:
				throw std::runtime_error( "ARKode failed with error " + std::to_string( errorFlag ) );
			break;
		}

		// ARKode has evolved us to t = tRet, update the plasma object and write it out.
		plasma.SetTime( tRet );
		plasma.ElectronTemperature = ELECTRON_TEMPERATURE( initialCondition );
		plasma.IonTemperature = ION_TEMPERATURE( initialCondition );
		plasma.SetMachFromVoltage();
		plasma.UpdatePhi();
		plasma.ComputeSteadyStateNeutrals();
		plasma.WriteTimeslice( tRet );

#if defined( DEBUG )
		std::cerr << "Writing timeslice at t = " << tRet << std::endl;
#endif
#if defined( DEBUG ) && defined( SUNDIALS_DEBUG )
	std::cerr << "After evolving to " << tRet << " T_i = " << ION_TEMPERATURE( initialCondition ) << " ; T_e = " << ELECTRON_TEMPERATURE( initialCondition ) << std::endl;
#endif

		double RelativeIonRate = ::fabs( ( plasma.IonHeating() - plasma.IonHeatLosses() )/( plasma.IonDensity * plasma.IonTemperature * ReferenceTemperature * ReferenceDensity ) );
		double RelativeElectronRate =::fabs( ( plasma.ElectronHeating() - plasma.ElectronHeatLosses() )/( plasma.ElectronDensity * plasma.ElectronTemperature * ReferenceTemperature * ReferenceDensity ) );
#if defined( DEBUG ) && defined( SUNDIALS_DEBUG )
		std::cerr << " Relative Rate of Change in Ion Energy Density " << RelativeIonRate * 100 << " %/s" << std::endl;
		std::cerr << " Relative Rate of Change in Electron Energy Density " << RelativeElectronRate * 100 << " %/s" << std::endl;
#endif
		if ( !plasma.isTimeDependent &&
		     RelativeIonRate < plasma.RateThreshold &&
		     RelativeElectronRate < plasma.RateThreshold )
		{
#if defined( DEBUG )
	std::cerr << "Steady state reached at time " << tRet << " with T_i = " << ION_TEMPERATURE( initialCondition ) << " ; T_e = " << ELECTRON_TEMPERATURE( initialCondition ) << std::endl;
#endif
			break;
		}

	}

#ifdef DEBUG
	long nSteps = 0,nfeEvals = 0,nfiEvals = 0;
	ArkodeErrorWrapper( ARKodeGetNumSteps( arkMem, &nSteps ), "ARKGetNumSteps" );
	ArkodeErrorWrapper( ARKStepGetNumRhsEvals( arkMem, &nfeEvals, &nfiEvals ), "ARKGetNumRhsEvals" );
	std::cerr << "SUNDIALS Timestepping took " << nSteps << " internal timesteps resulting in " << nfiEvals << " implicit function evaluations" << std::endl;
#endif

	// Teardown 
	{
		SUNLinSolFree( LS );
		SUNMatDestroy( Jacobian );
		N_VDestroy( initialCondition );
		ARKodeFree( &arkMem );
	}
}

// e.g. Feedback Control of Voltage on Te
void MCTransConfig::doFixedTeSolve( MirrorPlasma& plasma ) const
{
	throw std::logic_error( "Fixed T_e solve not yet fully implemented" );
	sundials::Context sunctx;
	sunindextype NDims = N_DIMENSIONS;
	N_Vector initialCondition = N_VNew_Serial( NDims, sunctx );

	double InitialTemperature = plasma.InitialTemp;
	ION_TEMPERATURE( initialCondition ) = InitialTemperature;
	ELECTRON_TEMPERATURE( initialCondition ) = InitialTemperature;

	realtype t0 = 0;

	void *arkMem = ARKStepCreate( nullptr, ARKode_TemperatureSolve, t0, initialCondition, sunctx );

	if ( arkMem == nullptr ) {
		throw std::runtime_error( "Cannot allocate ARKode Working Memory" );
	}

	// Dummy Jacobian, will be filled by ARKode with finite-difference approximations
	SUNMatrix       Jacobian = SUNDenseMatrix( NDims, NDims, sunctx );
	// Small system, direct solve is fastest
	SUNLinearSolver  LS = SUNLinSol_Dense( initialCondition, Jacobian, sunctx );

	ArkodeErrorWrapper( ARKodeSetLinearSolver( arkMem, LS, Jacobian ), "ARKodeSetLinearSolver" );
	
	

	double abstol = plasma.SundialsAbsTol;
	double reltol = plasma.SundialsRelTol;

	ArkodeErrorWrapper( ARKodeSStolerances( arkMem, reltol, abstol ), "ARKodeSStolerances" );
	ArkodeErrorWrapper( ARKStepSetTableNum( arkMem, IRK_SCHEME, ARKSTEP_NULL_STEPPER ), "ARKodeSetTableNum" );
	
	ArkodeErrorWrapper( ARKodeSetUserData( arkMem, reinterpret_cast<void*>( &plasma ) ), "ARKodeSetUserData" );


	const unsigned long MaxSteps = 1e4;
	ArkodeErrorWrapper( ARKodeSetMaxNumSteps( arkMem, MaxSteps ), "ARKodeSetMaxNumSteps" );

	realtype t,tRet;	
	int errorFlag = ARKodeEvolve( arkMem, t, initialCondition, &tRet, ARK_NORMAL );
	switch ( errorFlag ) {
		case ARK_SUCCESS:
			break;
		default:
			throw std::runtime_error( "KINSol failed with error " + std::to_string( errorFlag ) );
			break;
	}	

	// We've solved and found the answer. Update the plasma object

	plasma.ElectronTemperature = ELECTRON_TEMPERATURE( initialCondition );
	plasma.IonTemperature      =      ION_TEMPERATURE( initialCondition );

	// Teardown 
	{
		SUNLinSolFree( LS );
		SUNMatDestroy( Jacobian );
		N_VDestroy( initialCondition );
		ARKodeFree( &arkMem );
	}

	plasma.SetMachFromVoltage();
	plasma.ComputeSteadyStateNeutrals();
}

