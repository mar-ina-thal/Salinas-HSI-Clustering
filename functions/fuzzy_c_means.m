function [ U, C ] = fuzzy_c_means( inputs, nClusters, fuzzification, error )
%% FUZZY_C_MEANS 
% 
% This function execute fuzzy c-means algorithm over a input data
% 
% INPUT ARGUMENTS
%
% 'inputs'        | (matrix) input data to be clustered.
%                   the data should be organized as a matrix rows, where
%                   colunms represents the samples of each input (row)
% 'nClusters'     | (scalar) numbers of clusters 
% 'fuzzification' | (scalar) is the fuzzification parameter, usually in the
%                   range [1.25, 2]
% 'error'         | (scalar) minimum error that the algorithm must reach
% 
% OUTPUT ARGUMENTS
% 
% 'U'             | (matrix) membership matrix, contains the membership
%                   values of the inputs to the each cluster, where the
%                   number of the column represent the number of the
%                   cluster
% 'C'             | (matrix) cluter's centroids points, each row represent
%                   the centroid points

%% EXAMPLE 1
% 2D Data
%
% sample1 = [ ( 75.2 - 32 ).*rand( 1, 20 ) + 32, ...
%            ( 60 - 20 ).*rand( 1,20 ) + 20, ...
%            ( 15 -( -3.4 ) ).*rand( 1, 20 ) + ( -3.4 ) 
%           ]';
% 
% sample2 = [ ( 19.2 - 1 ).*rand( 1,20 ) + 1, ...
%            ( -10 - ( -42.5 ) ).*rand( 1,20 ) + ( -42.5 ), ...
%            ( 20.4 - ( -5 ) ).*rand( 1,20 ) + ( -5 ) 
%           ]';
% 
% inputs = [ sample1, sample2 ];
% 
% [ U, C ] = fuzzy_c_means( inputs, 3, 2, 1e-9 );


%% EXAMPLE 2
% 3D Data
%
% sample1 = [ ( 75.2 - 32 ).*rand( 1, 20 ) + 32, ...
%            ( 60 - 20 ).*rand( 1,20 ) + 20, ...
%            ( 15 -( -3.4 ) ).*rand( 1, 20 ) + ( -3.4 ) 
%           ]';
% 
% sample2 = [ ( 19.2 - 1 ).*rand( 1,20 ) + 1, ...
%            ( -10 - ( -42.5 ) ).*rand( 1,20 ) + ( -42.5 ), ...
%            ( 20.4 - ( -5 ) ).*rand( 1,20 ) + ( -5 ) 
%           ]';
% 
% sample3 = [ ( 100 - 70 ).*rand( 1, 20 ) + 70, ...
%            ( 45 - 30 ).*rand( 1,20 ) + 30, ...
%            ( 60 - 40 ).*rand( 1,20 ) + 40
%           ]';
% 
% inputs = [ sample1, sample2, sample3 ];
% 
% [ U, C ] = fuzzy_c_means( inputs, 3, 2, 1e-9 );


%% RUN

    [nInputs, nInputSamples ] = size( inputs );

    % Number of Clusters
    nC = nClusters;
    % fuzzification parameter
    m = fuzzification; 

    % Generate the random association of inputs to the clusters, values [ 0, 1 ]
    n = rand( nInputs, nC );
    nTotal = sum( n, 2 );
    RandomAssociationValues = ( n./nTotal );

    % Initial Membership Matrix
    U = RandomAssociationValues;
    % Cluster's centroids
    C = zeros( nC, nInputSamples );

    % Aux Parameters
    t = 0;
    currentError = 1;
    ilustrate = 0;
    
    % plot parameters
    minAxis = min( inputs );
    maxAxis = max( inputs );
    
    %--- visualize data when it's possible
    if ismatrix( inputs ) && ( nInputSamples == 2 || nInputSamples == 3 )
        fig = figure( 'Name', 'Fuzzy c Means Ilustration', 'NumberTitle', 'off' );
        if nInputSamples == 2
            ilustrate = 1;
            plot2DData( fig, 1, inputs, 0, minAxis, maxAxis, 'Input Data' )
        elseif nInputSamples == 3
            ilustrate = 1;
            fig2 = figure( 'Name', 'Fuzzy c Means Ilustration 1', 'NumberTitle', 'off' );
            plot3DData( fig, inputs, 0, 'Input Data')
        end
    end

    while currentError > error 
        U0 = U;

        % Calculate the cluster's centroids
        for i = 1 : 1 : nC
            for j = 1 : 1 : nInputSamples
                C( i, j ) = ( sum ( inputs( :, j ) .* ( U( :, i ).^m ) ) )/( sum ( U( :, i ).^m ) );
            end
        end
        

        % calculate dissimilarly between the inputs and centroids using
        % euclidean distance
        distanceFromCluster = zeros( nInputs, nC );
        for k = 1 : 1 : nC
            distance = sum( ( ( inputs - C( k, : ) ).^2 ), 2 );
            distanceFromCluster( : , k) = sqrt( distance );
        end

        % update membership matrix values
        den = sum( ( ( 1./distanceFromCluster ).^( 1/( m-1 ) ) ), 2 );
        for z = 1 : 1 : nC
            num = ( ( 1./distanceFromCluster( :, z ) ).^( 1/( m-1 ) ) ) ./ den;
            U( :, z ) = num';
        end
        
        currentError = ( sqrt( ( U - U0 ).^2 ) );
        
        t = t + 1;
        if ilustrate
            if nInputSamples == 2
                plot2DData( fig, 2, inputs, C, minAxis, maxAxis, [ 'Centroids at ', 'Epoch: ', num2str( t ) ] );
            elseif nInputSamples == 3
                plot3DData( fig2, inputs, C, [ 'Centroids at ', 'Epoch: ', num2str( t ) ] );
            end
        end
    end
    if ilustrate
        plotClusterData( inputs, U, nC );
    end


function plot2DData( fig, nSubplot, data, centroid, minAxis, maxAxis, description )
    figure( fig );
    np = subplot( 1, 2, nSubplot );
    if centroid
        cla( np );
        hold on;
        plot( data( :, 1 ), data( :, 2 ), 'ro' );
        plot( centroid( :, 1 ), centroid( :, 2 ), 'kx', 'LineWidth', 3, 'MarkerSize', 15 );
        hold off;
    else
        plot( data( :, 1 ), data( :, 2 ), 'ro' );
    end
    if exist( 'description', 'var')
        title( description );
    end
    if exist( 'minAxis', 'var') && exist( 'maxAxis', 'var')
        width = 10;
        axis( [ minAxis( 1 ) + ( ( minAxis( 1 )*-1 )/abs( minAxis( 1 ) ) )*-width, ...
            maxAxis( 1 ) + ( ( maxAxis( 1 )*-1 )/abs( maxAxis( 1 ) ) )*-width, ...
            minAxis( 2 ) + ( ( minAxis( 2 )*-1 )/abs( minAxis( 2 ) ) )*-width, ...
            maxAxis( 2 ) + ( ( maxAxis( 2 )*-1 )/abs( maxAxis( 2 ) ) )*-width, ...
        ] )
    else
        axis square;
    end
    

function plot3DData( fig, data, centroid, description )
    figure( fig );
    clf(fig);
    if centroid
        plot3( data( :, 1 ), data( :, 2 ), data( :, 3 ), 'ro');        
        hold on;
        plot3( centroid( :, 1 ), centroid( :, 2 ), centroid( :, 3 ), 'kx', 'LineWidth', 3, 'MarkerSize', 15);
        hold off;
    else
        plot3( data( :, 1 ), data( :, 2 ), data( :, 3 ), 'ro');
    end
    if exist( 'description', 'var')
        title( description );
    end
    axis square;
    grid on;

function plotClusterData ( inputs, U, nC )
    colors = rand( nC, 3 );
    [ nInput, samples ] = size( inputs );
    figure( 'Name', 'Classified Data', 'NumberTitle', 'off' );
    for n = 1 : 1 : nInput
        [ ~, c ] = max( U( n, : ) );
        if samples == 3
            plot3( inputs( n, 1 ), inputs( n, 2 ), inputs( n, 3 ), 'Color', colors( c, : ), 'Marker', 'o' );
        else
            plot( inputs( n, 1 ), inputs( n, 2 ), 'Color', colors( c, : ), 'Marker', 'o' );
        end
        hold on;
    end
    if samples == 3
        grid on;
    end
    title( 'Classified Data' );